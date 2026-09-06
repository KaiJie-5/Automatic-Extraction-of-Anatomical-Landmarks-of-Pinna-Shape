"""Screen or evaluate a bilateral mean/asymmetry PCA prior on one fold."""

from __future__ import annotations

import argparse
import itertools
from typing import Mapping, Sequence

import numpy as np

from ..dataset import Dataset
from ..pipeline_dataset import EAR_NAMES, prediction_key, prepare_ear_geometry
from ..surface import project_points_to_mesh
from .bilateral_pca import (
    CONTOUR_RANGES,
    BilateralMeanAsymmetryPCAPrior,
    gate_bilateral_contours,
    normalise_contour_gate,
)
from .evaluate_prior import (
    _device,
    _distribution,
    _load_model,
    _part_means,
    _predict_bilateral_local,
    _predict_local,
    _validate_context,
    _world_from_local,
    _write_json,
)
from .fitting import file_sha256, load_center_predictions, read_json
from .generate_prior import positive_int, unit_float
from .pca import PCAShapePrior


def _unique(values: Sequence[int | float]) -> list[int | float]:
    return list(dict.fromkeys(values))


def _setting_id(
    setting: Mapping[str, int | float], contour_gate: Sequence[str] = ()
) -> str:
    def number(value: float) -> str:
        return f"{float(value):g}".replace(".", "p")

    identifier = (
        f"common_c{setting['common_components']}_b{number(setting['common_beta'])}_"
        f"asym_c{setting['asymmetry_components']}_b"
        f"{number(setting['asymmetry_beta'])}"
    )
    if contour_gate:
        identifier += "_gate_" + "+".join(contour_gate)
    return identifier


def _load_independent_reference_prior(args, reference) -> PCAShapePrior | None:
    if not args.contour_gate:
        if args.independent_prior_path is not None:
            raise ValueError(
                "--independent-prior-path is only valid with --contour-gate"
            )
        return None
    if args.independent_prior_path is None:
        raise ValueError(
            "--contour-gate requires --independent-prior-path so the unchanged "
            "contours exactly reproduce the reference pipeline"
        )
    if reference.get("prior_sha256") != file_sha256(args.independent_prior_path):
        raise ValueError(
            "independent prior hash does not match the reference report"
        )
    prior = PCAShapePrior.load(args.independent_prior_path)
    components = int(reference.get("components", -1))
    beta = float(reference.get("beta", float("nan")))
    if components <= 0 or components > len(prior.components):
        raise ValueError("reference report has an invalid independent PCA component count")
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("reference report has an invalid independent PCA beta")
    return prior


def _validate_prior_manifest(
    manifest: Mapping[str, object],
    args,
    outer_fold: int,
    training_ids: Sequence[str],
    prior: BilateralMeanAsymmetryPCAPrior,
) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("prior_type") != "bilateral_mean_asymmetry_pca"
    ):
        raise ValueError("manifest is not a bilateral mean/asymmetry PCA manifest")
    if manifest.get("mode") != "fold" or str(manifest.get("outer_fold")) != str(
        outer_fold
    ):
        raise ValueError("bilateral PCA manifest does not match checkpoint fold")
    if list(manifest.get("train_subject_ids", [])) != list(training_ids):
        raise ValueError(
            "bilateral PCA training IDs do not exactly match outer-training fold"
        )
    expected_hashes = {
        "folds_json_sha256": file_sha256(args.folds_json),
        "predictions_json_sha256": file_sha256(args.predictions_json),
        "calibration_json_sha256": file_sha256(args.calibration_json),
        "prior_sha256": file_sha256(args.prior_path),
    }
    mismatches = [
        key for key, value in expected_hashes.items() if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "bilateral PCA manifest hash mismatch: " + ", ".join(mismatches)
        )
    if int(manifest.get("common_components", -1)) != len(
        prior.common_components
    ) or int(manifest.get("asymmetry_components", -1)) != len(
        prior.asymmetry_components
    ):
        raise ValueError("bilateral PCA manifest component counts do not match prior")
    if max(args.common_components) > len(prior.common_components):
        raise ValueError("requested common components exceed the saved prior")
    if max(args.asymmetry_components) > len(prior.asymmetry_components):
        raise ValueError("requested asymmetry components exceed the saved prior")


def _validate_reference(
    report: Mapping[str, object],
    checkpoint_path: str,
    outer_fold: int,
    run_seed: int,
    validation_ids: Sequence[str],
) -> None:
    if (
        report.get("schema_version") != 2
        or report.get("component") != "fold_projection_aware_pca_evaluation"
    ):
        raise ValueError("reference report must be an independent PCA fold report")
    if int(report.get("outer_fold", -1)) != outer_fold or int(
        report.get("run_seed", -1)
    ) != run_seed:
        raise ValueError("reference report fold/seed does not match checkpoint")
    if report.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError("reference report was not generated from this checkpoint")
    subjects = report.get("subject_level_raw_and_pca_errors")
    if not isinstance(subjects, Mapping) or set(subjects) != set(validation_ids):
        raise ValueError("reference report held-out subjects do not match checkpoint")
    for subject_id in validation_ids:
        ears = subjects[subject_id]
        if not isinstance(ears, Mapping) or set(ears) != set(EAR_NAMES):
            raise ValueError(f"invalid reference ear rows for {subject_id}")
        for ear in EAR_NAMES:
            row = ears[ear]
            for field in (
                "raw_md_mm",
                "pca_md_mm",
                "projected_md_mm",
                "pca_projected_md_mm",
            ):
                value = float(row[field])
                if not np.isfinite(value):
                    raise ValueError(f"non-finite reference {field} for {subject_id}:{ear}")


def _candidate_summary(candidate: Mapping[str, object], projected: bool) -> dict:
    pca_stack = np.stack(candidate["pca_errors"])
    pca_ear = pca_stack.mean(axis=1)
    subject_rows = candidate["subjects"]
    reference_pca_ear = np.asarray(
        [
            float(subject_rows[subject_id][ear]["reference_pca_md_mm"])
            for subject_id in subject_rows
            for ear in EAR_NAMES
        ],
        dtype=np.float32,
    )
    result = {
        "setting_id": candidate["setting_id"],
        **candidate["setting"],
        "pca_mean_md_mm": float(pca_ear.mean()),
        "pca_per_part_md_mm": _part_means(pca_stack),
        "pca_distribution_mm": _distribution(pca_ear),
        "reference_pca_mean_md_mm": float(reference_pca_ear.mean()),
        "pca_improvement_vs_reference_mm": float(
            reference_pca_ear.mean() - pca_ear.mean()
        ),
        "pca_ears_improved_vs_reference": int(
            np.sum(pca_ear < reference_pca_ear)
        ),
        "pca_ears_worsened_vs_reference": int(
            np.sum(pca_ear > reference_pca_ear)
        ),
    }
    if projected:
        projected_stack = np.stack(candidate["projected_errors"])
        projected_ear = projected_stack.mean(axis=1)
        reference_projected_ear = np.asarray(
            [
                float(
                    subject_rows[subject_id][ear][
                        "reference_pca_projected_md_mm"
                    ]
                )
                for subject_id in subject_rows
                for ear in EAR_NAMES
            ],
            dtype=np.float32,
        )
        result.update(
            pca_projected_mean_md_mm=float(projected_ear.mean()),
            pca_projected_per_part_md_mm=_part_means(projected_stack),
            pca_projected_distribution_mm=_distribution(projected_ear),
            reference_pca_projected_mean_md_mm=float(
                reference_projected_ear.mean()
            ),
            pca_projected_improvement_vs_reference_mm=float(
                reference_projected_ear.mean() - projected_ear.mean()
            ),
            pca_projected_ears_improved_vs_reference=int(
                np.sum(projected_ear < reference_projected_ear)
            ),
            pca_projected_ears_worsened_vs_reference=int(
                np.sum(projected_ear > reference_projected_ear)
            ),
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prior-path", required=True)
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--reference-report", required=True)
    parser.add_argument(
        "--independent-prior-path",
        help=(
            "independent-ear PCA artifact recorded by the reference report; "
            "required only for contour-gated evaluation"
        ),
    )
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument(
        "--common-components", nargs="+", type=positive_int, required=True
    )
    parser.add_argument(
        "--asymmetry-components", nargs="+", type=positive_int, required=True
    )
    parser.add_argument("--common-betas", nargs="+", type=unit_float, required=True)
    parser.add_argument(
        "--asymmetry-betas", nargs="+", type=unit_float, required=True
    )
    parser.add_argument(
        "--contour-gate",
        nargs="+",
        choices=tuple(CONTOUR_RANGES),
        help=(
            "use bilateral PCA only for these contours and retain the exact "
            "independent-PCA reference prediction elsewhere"
        ),
    )
    parser.add_argument(
        "--skip-projection",
        action="store_true",
        help="screen a broad PCA grid without expensive triangle projection",
    )
    parser.add_argument("--run-seed", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.common_components = [int(v) for v in _unique(args.common_components)]
    args.asymmetry_components = [
        int(v) for v in _unique(args.asymmetry_components)
    ]
    args.common_betas = [float(v) for v in _unique(args.common_betas)]
    args.asymmetry_betas = [float(v) for v in _unique(args.asymmetry_betas)]
    args.contour_gate = normalise_contour_gate(args.contour_gate)
    settings = [
        {
            "common_components": common_components,
            "asymmetry_components": asymmetry_components,
            "common_beta": common_beta,
            "asymmetry_beta": asymmetry_beta,
        }
        for common_components, asymmetry_components, common_beta, asymmetry_beta in itertools.product(
            args.common_components,
            args.asymmetry_components,
            args.common_betas,
            args.asymmetry_betas,
        )
    ]
    device = _device(args.device)
    _, model, model_config, data_config = _load_model(args.checkpoint_path, device)
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    subject_index = {
        dataset.get_identifier(index): index for index in range(len(dataset))
    }
    validation_ids, run_seed, num_points = _validate_context(
        args, data_config, dataset
    )
    calibration = read_json(args.calibration_json)
    predictions = load_center_predictions(
        args.predictions_json,
        [dataset.get_identifier(index) for index in range(len(dataset))],
    )
    prior = BilateralMeanAsymmetryPCAPrior.load(args.prior_path)
    prior_manifest = read_json(args.prior_manifest)
    _validate_prior_manifest(
        prior_manifest,
        args,
        int(data_config["outer_fold"]),
        list(data_config["train_ids"]),
        prior,
    )
    reference = read_json(args.reference_report)
    _validate_reference(
        reference,
        args.checkpoint_path,
        int(data_config["outer_fold"]),
        run_seed,
        validation_ids,
    )
    independent_prior = _load_independent_reference_prior(args, reference)

    candidates = [
        {
            "setting": setting,
            "setting_id": _setting_id(setting, args.contour_gate),
            "pca_errors": [],
            "projected_errors": [],
            "subjects": {},
        }
        for setting in settings
    ]
    raw_errors = []
    raw_projected_errors = []
    reference_subjects = reference["subject_level_raw_and_pca_errors"]
    backbone = str(model_config.get("backbone", ""))
    bilateral_mode = str(model_config.get("bilateral_mode", "none"))

    for subject_position, subject_id in enumerate(validation_ids):
        mesh, left, right = dataset[subject_index[subject_id]]
        ground_truth_by_ear = {"left": left, "right": right}
        prepared_by_ear = {}
        for ear_index, ear in enumerate(EAR_NAMES):
            item = subject_position * len(EAR_NAMES) + ear_index
            prepared_by_ear[ear] = prepare_ear_geometry(
                mesh,
                ground_truth_by_ear[ear],
                ear,
                predictions[prediction_key(subject_id, ear)],
                calibration,
                num_points,
                run_seed + 100_000 + item * 1009,
            )
        if bilateral_mode != "none":
            raw_by_ear = _predict_bilateral_local(
                model, model_config, prepared_by_ear, device
            )
        else:
            raw_by_ear = {
                ear: _predict_local(
                    model,
                    model_config,
                    backbone,
                    prepared_by_ear[ear],
                    ear,
                    device,
                )
                for ear in EAR_NAMES
            }
        raw_pair = np.stack([raw_by_ear[ear] for ear in EAR_NAMES], axis=0)
        bilateral_pairs = [
            prior.blend_pair(raw_pair, **candidate["setting"])
            for candidate in candidates
        ]
        independent_pair = None
        if independent_prior is not None:
            independent_pair = independent_prior.blend(
                raw_pair,
                beta=float(reference["beta"]),
                n_components=int(reference["components"]),
            )
            candidate_pairs = [
                gate_bilateral_contours(
                    independent_pair, bilateral_pair, args.contour_gate
                )
                for bilateral_pair in bilateral_pairs
            ]
        else:
            candidate_pairs = bilateral_pairs

        for ear_index, ear in enumerate(EAR_NAMES):
            prepared = prepared_by_ear[ear]
            ground_truth = ground_truth_by_ear[ear]
            raw_world = _world_from_local(prepared, ear, raw_pair[ear_index])
            raw_error = np.linalg.norm(raw_world - ground_truth, axis=1).astype(
                np.float32
            )
            raw_errors.append(raw_error)
            reference_row = reference_subjects[subject_id][ear]
            if not np.isclose(
                float(raw_error.mean()),
                float(reference_row["raw_md_mm"]),
                atol=1e-4,
                rtol=0.0,
            ):
                raise ValueError(
                    f"reproduced raw MD does not match reference for {subject_id}:{ear}"
                )
            independent_world = None
            if independent_pair is not None:
                independent_world = _world_from_local(
                    prepared, ear, independent_pair[ear_index]
                )
                independent_error = np.linalg.norm(
                    independent_world - ground_truth, axis=1
                ).astype(np.float32)
                if not np.isclose(
                    float(independent_error.mean()),
                    float(reference_row["pca_md_mm"]),
                    atol=1e-4,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "reproduced independent PCA MD does not match reference "
                        f"for {subject_id}:{ear}"
                    )
            raw_projected_error = None
            if not args.skip_projection:
                raw_projected = project_points_to_mesh(
                    raw_world, prepared.crop_mesh
                )
                raw_projected_error = np.linalg.norm(
                    raw_projected - ground_truth, axis=1
                ).astype(np.float32)
                raw_projected_errors.append(raw_projected_error)
                if not np.isclose(
                    float(raw_projected_error.mean()),
                    float(reference_row["projected_md_mm"]),
                    atol=1e-4,
                    rtol=0.0,
                ):
                    raise ValueError(
                        "reproduced projected MD does not match reference for "
                        f"{subject_id}:{ear}"
                    )
                if independent_world is not None:
                    independent_projected = project_points_to_mesh(
                        independent_world, prepared.crop_mesh
                    )
                    independent_projected_error = np.linalg.norm(
                        independent_projected - ground_truth, axis=1
                    ).astype(np.float32)
                    if not np.isclose(
                        float(independent_projected_error.mean()),
                        float(reference_row["pca_projected_md_mm"]),
                        atol=1e-4,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            "reproduced independent PCA projected MD does not "
                            f"match reference for {subject_id}:{ear}"
                        )

            for candidate_index, candidate in enumerate(candidates):
                local = candidate_pairs[candidate_index][ear_index]
                world = _world_from_local(prepared, ear, local)
                errors = np.linalg.norm(world - ground_truth, axis=1).astype(
                    np.float32
                )
                candidate["pca_errors"].append(errors)
                row = candidate["subjects"].setdefault(subject_id, {}).setdefault(
                    ear, {}
                )
                row.update(
                    raw_md_mm=float(raw_error.mean()),
                    pca_md_mm=float(errors.mean()),
                    reference_pca_md_mm=float(reference_row["pca_md_mm"]),
                )
                if not args.skip_projection:
                    projected = project_points_to_mesh(world, prepared.crop_mesh)
                    projected_errors = np.linalg.norm(
                        projected - ground_truth, axis=1
                    ).astype(np.float32)
                    candidate["projected_errors"].append(projected_errors)
                    row.update(
                        projected_md_mm=float(raw_projected_error.mean()),
                        pca_projected_md_mm=float(projected_errors.mean()),
                        reference_pca_projected_md_mm=float(
                            reference_row["pca_projected_md_mm"]
                        ),
                    )

    projected = not args.skip_projection
    summaries = [_candidate_summary(candidate, projected) for candidate in candidates]
    selection_field = (
        "pca_projected_mean_md_mm" if projected else "pca_mean_md_mm"
    )
    selected_index = min(
        range(len(summaries)),
        key=lambda index: (
            float(summaries[index][selection_field]),
            int(summaries[index]["common_components"])
            + int(summaries[index]["asymmetry_components"]),
            summaries[index]["setting_id"],
        ),
    )
    selected = summaries[selected_index]
    selected_candidate = candidates[selected_index]
    raw_stack = np.stack(raw_errors)
    raw_ear = raw_stack.mean(axis=1)
    reference_field = (
        "pca_projected_mean_md_mm" if projected else "pca_mean_md_mm"
    )
    reference_mean = float(reference[reference_field])
    candidate_mean = float(selected[selection_field])
    report = {
        "schema_version": 1,
        "component": "fold_bilateral_mean_asymmetry_pca_evaluation",
        "outer_fold": int(data_config["outer_fold"]),
        "run_seed": int(run_seed),
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_sha256": file_sha256(args.checkpoint_path),
        "prior_path": str(args.prior_path),
        "prior_sha256": file_sha256(args.prior_path),
        "prior_manifest_path": str(args.prior_manifest),
        "reference_report": str(args.reference_report),
        "reference_report_sha256": file_sha256(args.reference_report),
        "contour_gate": list(args.contour_gate),
        "postprocess_mode": (
            "contour_gated_bilateral_over_independent_pca"
            if args.contour_gate
            else "full_bilateral_pca"
        ),
        "subject_count": len(validation_ids),
        "ear_count": len(raw_errors),
        "candidate_count": len(candidates),
        "projection_evaluated": projected,
        "selection_metric": selection_field,
        "selected_setting": selected_candidate["setting"],
        "selected_setting_id": selected_candidate["setting_id"],
        "reference_mean_md_mm": reference_mean,
        "selected_mean_md_mm": candidate_mean,
        "improvement_vs_reference_mm": reference_mean - candidate_mean,
        "raw_mean_md_mm": float(raw_ear.mean()),
        "raw_per_part_md_mm": _part_means(raw_stack),
        "raw_distribution_mm": _distribution(raw_ear),
        "candidates": summaries,
        "subject_level_raw_and_pca_errors": selected_candidate["subjects"],
        **selected,
    }
    if independent_prior is not None:
        report.update(
            independent_prior_path=str(args.independent_prior_path),
            independent_prior_sha256=file_sha256(args.independent_prior_path),
            independent_components=int(reference["components"]),
            independent_beta=float(reference["beta"]),
        )
    if projected:
        projected_stack = np.stack(raw_projected_errors)
        projected_ear = projected_stack.mean(axis=1)
        report.update(
            projected_mean_md_mm=float(projected_ear.mean()),
            projected_per_part_md_mm=_part_means(projected_stack),
            projected_distribution_mm=_distribution(projected_ear),
        )
    _write_json(args.output, report)
    print(
        f"Evaluated {len(candidates)} bilateral PCA setting(s); "
        f"selected {report['selected_setting_id']} by {selection_field}"
    )
    print(
        f"Reference {reference_mean:.6f} mm, selected {candidate_mean:.6f} mm, "
        f"improvement {report['improvement_vs_reference_mm']:.6f} mm"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
