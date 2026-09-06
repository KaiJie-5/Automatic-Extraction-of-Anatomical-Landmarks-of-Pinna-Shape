"""Summarize five-fold/three-seed bilateral PCA confirmation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .summarize_prior import DEFAULT_SEEDS, FOLDS, _distribution


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ear_values(report: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    reference = []
    candidate = []
    subjects = report.get("subject_level_raw_and_pca_errors")
    if not isinstance(subjects, Mapping):
        raise ValueError("bilateral PCA report is missing subject-level metrics")
    for subject_id in sorted(subjects):
        ears = subjects[subject_id]
        if not isinstance(ears, Mapping) or set(ears) != {"left", "right"}:
            raise ValueError(f"invalid bilateral PCA rows for {subject_id}")
        for ear in ("left", "right"):
            row = ears[ear]
            reference.append(float(row["reference_pca_projected_md_mm"]))
            candidate.append(float(row["pca_projected_md_mm"]))
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if (
        not len(reference_array)
        or reference_array.shape != candidate_array.shape
        or not np.isfinite(reference_array).all()
        or not np.isfinite(candidate_array).all()
    ):
        raise ValueError("bilateral PCA report contains invalid ear metrics")
    return reference_array, candidate_array


def summarize(report_root: str | Path, seeds: Sequence[int]) -> dict:
    root = Path(report_root)
    configurations = set()
    fold_values = {fold: [[], []] for fold in FOLDS}
    all_reference = []
    all_candidate = []
    reports = []
    for fold in FOLDS:
        for seed in seeds:
            path = root / f"fold{fold}_seed{int(seed)}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing bilateral PCA confirmation report: {path}"
                )
            report = _read_json(path)
            if (
                report.get("schema_version") != 1
                or report.get("component")
                != "fold_bilateral_mean_asymmetry_pca_evaluation"
                or not bool(report.get("projection_evaluated"))
                or int(report.get("outer_fold", -1)) != fold
                or int(report.get("run_seed", -1)) != int(seed)
                or int(report.get("candidate_count", -1)) != 1
            ):
                raise ValueError(
                    f"bilateral PCA report identity does not match path: {path}"
                )
            setting = report.get("selected_setting")
            if not isinstance(setting, Mapping):
                raise ValueError(f"bilateral PCA setting missing from {path}")
            configurations.add(
                (
                    int(setting["common_components"]),
                    int(setting["asymmetry_components"]),
                    float(setting["common_beta"]),
                    float(setting["asymmetry_beta"]),
                    tuple(str(value) for value in report.get("contour_gate", [])),
                    (
                        int(report["independent_components"])
                        if report.get("contour_gate")
                        else None
                    ),
                    (
                        float(report["independent_beta"])
                        if report.get("contour_gate")
                        else None
                    ),
                )
            )
            reference, candidate = _ear_values(report)
            if int(report.get("ear_count", -1)) != len(reference):
                raise ValueError(f"ear count mismatch in {path}")
            if not np.isclose(
                float(report["reference_mean_md_mm"]), reference.mean(), atol=1e-6
            ):
                raise ValueError(f"reference mean mismatch in {path}")
            if not np.isclose(
                float(report["pca_projected_mean_md_mm"]),
                candidate.mean(),
                atol=1e-6,
            ):
                raise ValueError(f"candidate mean mismatch in {path}")
            fold_values[fold][0].append(reference)
            fold_values[fold][1].append(candidate)
            all_reference.append(reference)
            all_candidate.append(candidate)
            reports.append(str(path))
    if len(configurations) != 1:
        raise ValueError(
            "all bilateral PCA reports must use one locked PCA setting and "
            "contour gate"
        )
    (
        common_components,
        asymmetry_components,
        common_beta,
        asymmetry_beta,
        contour_gate,
        independent_components,
        independent_beta,
    ) = next(iter(configurations))
    reference_pooled = np.concatenate(all_reference)
    candidate_pooled = np.concatenate(all_candidate)
    per_fold = {}
    improved_folds = 0
    for fold in FOLDS:
        reference = np.concatenate(fold_values[fold][0])
        candidate = np.concatenate(fold_values[fold][1])
        improvement = float(reference.mean() - candidate.mean())
        improved_folds += int(improvement > 0.0)
        per_fold[str(fold)] = {
            "reference_independent_pca_projected_md_mm": float(reference.mean()),
            "bilateral_pca_projected_md_mm": float(candidate.mean()),
            "improvement_mm": improvement,
            "ear_evaluations": int(len(reference)),
        }
    pooled_improvement = float(reference_pooled.mean() - candidate_pooled.mean())
    return {
        "schema_version": 1,
        "component": "bilateral_mean_asymmetry_pca_confirmation_summary",
        "common_components": common_components,
        "asymmetry_components": asymmetry_components,
        "common_beta": common_beta,
        "asymmetry_beta": asymmetry_beta,
        "contour_gate": list(contour_gate),
        "postprocess_mode": (
            "contour_gated_bilateral_over_independent_pca"
            if contour_gate
            else "full_bilateral_pca"
        ),
        "independent_components": independent_components,
        "independent_beta": independent_beta,
        "folds": list(FOLDS),
        "seeds": [int(seed) for seed in seeds],
        "report_count": len(reports),
        "ear_evaluations": int(len(reference_pooled)),
        "reference_distribution_mm": _distribution(reference_pooled),
        "bilateral_pca_projected_distribution_mm": _distribution(
            candidate_pooled
        ),
        "pooled_improvement_mm": pooled_improvement,
        "improved_folds": int(improved_folds),
        "per_fold": per_fold,
        "promotion_rule": {
            "lower_pooled_15_run_md": bool(pooled_improvement > 0.0),
            "improve_at_least_three_folds": bool(improved_folds >= 3),
            "promoted": bool(pooled_improvement > 0.0 and improved_folds >= 3),
        },
        "reports": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = summarize(args.report_root, args.seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
