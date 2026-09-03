"""Leakage-safe re-decoding and oracle diagnostics for fold heatmap checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .canonical import decanonicalize_xyz
from .dataset import Dataset
from .pipeline_dataset import EAR_NAMES, prediction_key, prepare_ear_geometry
from .precision import checkpoint_autocast_context
from .proposal_models import PointNeXtSurfaceHeatmapRegressor
from .shape_prior.evaluate_prior import (
    _load_model,
    _validate_context,
    _validate_prior_manifest,
)
from .shape_prior.fitting import file_sha256, load_center_predictions, read_json
from .shape_prior.pca import PCAShapePrior
from .surface import project_points_to_mesh


CONTOURS = (
    ("outer_helix", 0, 25),
    ("concha", 25, 55),
    ("inner_helix", 55, 75),
    ("superior_antihelix", 75, 85),
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1]")
    return parsed


def _distribution(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or not len(finite) or not np.isfinite(finite).all():
        raise ValueError("metric distribution must be one-dimensional and finite")
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "maximum": float(np.max(finite)),
    }


def _pearson(first: Sequence[float], second: Sequence[float]) -> float | None:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if (
        x.shape != y.shape
        or x.ndim != 1
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or float(np.std(x)) <= 1e-12
        or float(np.std(y)) <= 1e-12
    ):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def heatmap_uncertainty(logits: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Return normalized entropy, maximum probability, and top-two margin."""
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    probabilities = torch.exp(log_probabilities)
    entropy = -(
        probabilities * log_probabilities
    ).sum(dim=-1) / max(math.log(max(int(logits.shape[-1]), 2)), 1.0)
    top = probabilities.topk(min(2, int(probabilities.shape[-1])), dim=-1).values
    peak = top[..., 0]
    margin = peak - top[..., 1] if top.shape[-1] == 2 else peak
    return {"entropy": entropy, "peak_probability": peak, "peak_margin": margin}


def _world_from_local(prepared, ear: str, local: np.ndarray) -> np.ndarray:
    canonical = prepared.transform.denormalize_xyz(local)
    return decanonicalize_xyz(canonical, ear).astype(np.float32)


def _errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.linalg.norm(
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(target, dtype=np.float64),
        axis=-1,
    )
    if values.shape != (85,) or not np.isfinite(values).all():
        raise RuntimeError("landmark errors must be finite with shape (85,)")
    return values.astype(np.float32)


def _metric_summary(error_rows: Sequence[np.ndarray]) -> dict:
    values = np.stack(error_rows).astype(np.float64)
    ear_md = values.mean(axis=1)
    return {
        "pooled_md_mm": float(values.mean()),
        "per_landmark_md_mm": values.mean(axis=0).tolist(),
        "per_contour_md_mm": {
            name: float(values[:, start:end].mean())
            for name, start, end in CONTOURS
        },
        "ear_distribution_mm": _distribution(ear_md),
    }


def _configuration_key(topk: int, temperature: float) -> str:
    temperature_text = f"{float(temperature):g}".replace(".", "p")
    return f"topk_{int(topk)}_temperature_{temperature_text}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prior-path", required=True)
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--top-k", nargs="+", type=_positive_int, required=True)
    parser.add_argument(
        "--temperatures", nargs="+", type=_positive_float, required=True
    )
    parser.add_argument("--components", type=_positive_int, default=32)
    parser.add_argument("--beta", type=_unit_float, default=0.5)
    parser.add_argument("--run-seed", type=int)
    parser.add_argument("--projection-workers", type=_positive_int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = _device(args.device)
    checkpoint, model, model_config, data_config = _load_model(
        args.checkpoint_path, device
    )
    del checkpoint
    if not isinstance(model, PointNeXtSurfaceHeatmapRegressor):
        raise ValueError(
            "analyze-heatmap-decoder requires a PointNeXt surface-heatmap checkpoint"
        )
    if str(model_config.get("decoder")) != "surface_heatmap":
        raise ValueError("checkpoint is not a surface-heatmap landmark model")

    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    dataset_ids = [dataset.get_identifier(index) for index in range(len(dataset))]
    subject_index = {subject_id: index for index, subject_id in enumerate(dataset_ids)}
    validation_ids, run_seed, num_points = _validate_context(
        args, data_config, dataset
    )
    top_k = sorted(set(int(value) for value in args.top_k))
    temperatures = sorted(set(float(value) for value in args.temperatures))
    if top_k[-1] > num_points:
        raise ValueError(
            f"requested top-k {top_k[-1]} exceeds checkpoint point count {num_points}"
        )
    grid = [(k, temperature) for k in top_k for temperature in temperatures]

    calibration = read_json(args.calibration_json)
    predictions = load_center_predictions(args.predictions_json, dataset_ids)
    prior = PCAShapePrior.load(args.prior_path)
    prior_manifest = read_json(args.prior_manifest)
    _validate_prior_manifest(
        prior_manifest,
        args,
        int(data_config["outer_fold"]),
        list(data_config["train_ids"]),
        prior,
    )
    if int(args.components) > int(prior.n_components):
        raise ValueError(
            f"requested {args.components} PCA components but prior stores "
            f"{prior.n_components}"
        )

    stage_errors = {
        name: []
        for name in (
            "coarse",
            "final",
            "final_pca",
            "final_projected",
            "final_pca_projected",
        )
    }
    grid_errors = {
        _configuration_key(k, temperature): {
            "final": [],
            "pca": [],
            "pca_projected": [],
        }
        for k, temperature in grid
    }
    oracle_sample_errors = []
    oracle_triangle_errors = []
    entropy_values = []
    peak_values = []
    margin_values = []
    final_error_values = []
    pca_projected_error_values = []
    per_ear = {}
    started = time.perf_counter()
    total = len(validation_ids) * len(EAR_NAMES)
    completed = 0

    for subject_position, subject_id in enumerate(validation_ids):
        mesh, left, right = dataset[subject_index[subject_id]]
        for ear_offset, ear in enumerate(EAR_NAMES):
            item = subject_position * 2 + ear_offset
            ground_truth = left if ear == "left" else right
            center = predictions[prediction_key(subject_id, ear)]
            prepared = prepare_ear_geometry(
                mesh,
                ground_truth,
                ear,
                center,
                calibration,
                num_points,
                run_seed + 100_000 + item * 1009,
            )
            points = torch.from_numpy(
                prepared.point_features.astype(np.float32)
            ).unsqueeze(0).to(device)
            local_grid = {}
            with torch.no_grad(), checkpoint_autocast_context(device, model_config):
                details = model.forward_with_details(points)
                logits = details["heatmap_logits"]
                candidates = details["surface_candidates"]
                point_features = details["decoded_point_features"]
                query_features = details["landmark_query_features"]
                uncertainty = heatmap_uncertainty(logits)
                for k, temperature in grid:
                    coarse = model.decode_surface_coordinates(
                        logits, candidates, topk=k, temperature=temperature
                    )
                    final, _ = model.apply_refinement(
                        coarse,
                        points,
                        point_features,
                        logits,
                        query_features,
                    )
                    local_grid[(k, temperature)] = (
                        final.squeeze(0).float().cpu().numpy().astype(np.float32)
                    )

            coarse_local = (
                details["coarse"].squeeze(0).float().cpu().numpy().astype(np.float32)
            )
            final_local = (
                details["final"].squeeze(0).float().cpu().numpy().astype(np.float32)
            )
            candidate_local = (
                candidates.squeeze(0).float().cpu().numpy().astype(np.float32)
            )
            entropy = uncertainty["entropy"].squeeze(0).cpu().numpy()
            peak = uncertainty["peak_probability"].squeeze(0).cpu().numpy()
            margin = uncertainty["peak_margin"].squeeze(0).cpu().numpy()

            coarse_world = _world_from_local(prepared, ear, coarse_local)
            final_world = _world_from_local(prepared, ear, final_local)
            pca_local = prior.blend(
                final_local,
                beta=float(args.beta),
                n_components=int(args.components),
            )
            pca_world = _world_from_local(prepared, ear, pca_local)
            final_projected = project_points_to_mesh(
                final_world, prepared.crop_mesh
            )
            pca_projected = project_points_to_mesh(
                pca_world, prepared.crop_mesh
            )
            stage_errors["coarse"].append(_errors(coarse_world, ground_truth))
            stage_errors["final"].append(_errors(final_world, ground_truth))
            stage_errors["final_pca"].append(_errors(pca_world, ground_truth))
            stage_errors["final_projected"].append(
                _errors(final_projected, ground_truth)
            )
            stage_errors["final_pca_projected"].append(
                _errors(pca_projected, ground_truth)
            )

            target_local = prepared.target.astype(np.float32)
            nearest = np.linalg.norm(
                target_local[:, None, :] - candidate_local[None, :, :], axis=-1
            ).min(axis=1) * float(calibration["local_scale"])
            oracle_sample_errors.append(nearest.astype(np.float32))
            triangle_oracle = project_points_to_mesh(
                np.asarray(ground_truth, dtype=np.float32), prepared.crop_mesh
            )
            oracle_triangle_errors.append(_errors(triangle_oracle, ground_truth))

            final_errors = stage_errors["final"][-1]
            selected_errors = stage_errors["final_pca_projected"][-1]
            entropy_values.extend(entropy.tolist())
            peak_values.extend(peak.tolist())
            margin_values.extend(margin.tolist())
            final_error_values.extend(final_errors.tolist())
            pca_projected_error_values.extend(selected_errors.tolist())

            ear_grid = {}
            pending_projection = {}
            for k, temperature in grid:
                key = _configuration_key(k, temperature)
                decoded_local = local_grid[(k, temperature)]
                decoded_world = _world_from_local(prepared, ear, decoded_local)
                decoded_pca_local = prior.blend(
                    decoded_local,
                    beta=float(args.beta),
                    n_components=int(args.components),
                )
                decoded_pca_world = _world_from_local(
                    prepared, ear, decoded_pca_local
                )
                final_error = _errors(decoded_world, ground_truth)
                pca_error = _errors(decoded_pca_world, ground_truth)
                grid_errors[key]["final"].append(final_error)
                grid_errors[key]["pca"].append(pca_error)
                pending_projection[key] = decoded_pca_world
                ear_grid[key] = {
                    "final_md_mm": float(final_error.mean()),
                    "pca_md_mm": float(pca_error.mean()),
                }

            def project_grid_item(item):
                key, values = item
                return key, project_points_to_mesh(values, prepared.crop_mesh)

            pending_items = list(pending_projection.items())
            if int(args.projection_workers) == 1:
                projected_items = map(project_grid_item, pending_items)
            else:
                executor = ThreadPoolExecutor(max_workers=args.projection_workers)
                projected_items = executor.map(project_grid_item, pending_items)
            try:
                for key, decoded_projected in projected_items:
                    projected_error = _errors(decoded_projected, ground_truth)
                    grid_errors[key]["pca_projected"].append(projected_error)
                    ear_grid[key]["pca_projected_md_mm"] = float(
                        projected_error.mean()
                    )
            finally:
                if int(args.projection_workers) != 1:
                    executor.shutdown(wait=True)

            per_ear[prediction_key(subject_id, ear)] = {
                "default_final_md_mm": float(final_errors.mean()),
                "default_pca_projected_md_mm": float(selected_errors.mean()),
                "sample_candidate_oracle_md_mm": float(nearest.mean()),
                "triangle_oracle_md_mm": float(
                    oracle_triangle_errors[-1].mean()
                ),
                "mean_normalized_entropy": float(np.mean(entropy)),
                "mean_peak_probability": float(np.mean(peak)),
                "mean_peak_margin": float(np.mean(margin)),
                "grid": ear_grid,
            }
            completed += 1
            print(f"Analyzed heatmap decoder {completed}/{total}: {subject_id}:{ear}")

    grid_summary = {}
    for k, temperature in grid:
        key = _configuration_key(k, temperature)
        grid_summary[key] = {
            "topk": int(k),
            "temperature": float(temperature),
            "final": _metric_summary(grid_errors[key]["final"]),
            "pca": _metric_summary(grid_errors[key]["pca"]),
            "pca_projected": _metric_summary(
                grid_errors[key]["pca_projected"]
            ),
        }
    selected_key = min(
        grid_summary,
        key=lambda key: (
            grid_summary[key]["pca_projected"]["pooled_md_mm"],
            grid_summary[key]["pca_projected"]["ear_distribution_mm"]["p95"],
            grid_summary[key]["topk"],
            grid_summary[key]["temperature"],
        ),
    )
    report = {
        "schema_version": 1,
        "component": "fold_surface_heatmap_decoder_diagnostic",
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_sha256": file_sha256(args.checkpoint_path),
        "prior_path": str(args.prior_path),
        "prior_sha256": file_sha256(args.prior_path),
        "prior_manifest_path": str(args.prior_manifest),
        "outer_fold": int(data_config["outer_fold"]),
        "run_seed": int(run_seed),
        "subject_count": len(validation_ids),
        "ear_count": total,
        "num_points": int(num_points),
        "components": int(args.components),
        "beta": float(args.beta),
        "projection_workers": int(args.projection_workers),
        "default_decode": {
            "topk": int(model.heatmap_topk),
            "temperature": float(model.heatmap_coordinate_temperature),
        },
        "stage_ablation": {
            name: _metric_summary(rows) for name, rows in stage_errors.items()
        },
        "oracle": {
            "sample_candidates": _metric_summary(oracle_sample_errors),
            "triangle_surface": _metric_summary(oracle_triangle_errors),
        },
        "uncertainty_error_correlations": {
            "normalized_entropy_vs_final_error": _pearson(
                entropy_values, final_error_values
            ),
            "normalized_entropy_vs_pca_projected_error": _pearson(
                entropy_values, pca_projected_error_values
            ),
            "peak_probability_vs_final_error": _pearson(
                peak_values, final_error_values
            ),
            "peak_probability_vs_pca_projected_error": _pearson(
                peak_values, pca_projected_error_values
            ),
            "peak_margin_vs_final_error": _pearson(
                margin_values, final_error_values
            ),
            "peak_margin_vs_pca_projected_error": _pearson(
                margin_values, pca_projected_error_values
            ),
        },
        "grid": grid_summary,
        "selected": {"key": selected_key, **grid_summary[selected_key]},
        "runtime_seconds": float(time.perf_counter() - started),
        "per_ear": per_ear,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    selected = report["selected"]
    print(
        "Selected "
        f"top-k={selected['topk']}, temperature={selected['temperature']:g}, "
        "PCA-projected MD="
        f"{selected['pca_projected']['pooled_md_mm']:.6f} mm"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
