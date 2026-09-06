"""Mandatory candidate-retrieval diagnostics before geodesic voting training."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .dataset import Dataset
from .geodesic import (
    cache_path,
    load_geodesic_cache_entry,
    sample_geodesic_distances_from_barycentric,
    validate_geodesic_manifest,
)
from .pipeline_dataset import EAR_NAMES, prediction_key, prepare_ear_geometry
from .precision import checkpoint_autocast_context
from .proposal_models import PointNeXtSurfaceHeatmapRegressor
from .shape_prior.evaluate_prior import _load_model, _validate_context
from .shape_prior.fitting import load_center_predictions, read_json


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


def _bounded_dot(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("normal dot threshold must be in [-1, 1]")
    return parsed


def _distribution(values: np.ndarray) -> dict:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("diagnostic distribution must be finite and nonempty")
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p90": float(np.percentile(data, 90.0)),
        "p95": float(np.percentile(data, 95.0)),
        "maximum": float(np.max(data)),
    }


def candidate_retrieval_metrics(
    logits: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    scale_mm: float,
    top_k: Sequence[int],
) -> tuple[np.ndarray, Mapping[int, np.ndarray]]:
    """Return stable heatmap rank of nearest sample and top-K oracle errors."""
    scores = np.asarray(logits, dtype=np.float64)
    xyz = np.asarray(candidates, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != 85:
        raise ValueError("diagnostic logits must have shape (85, N)")
    if xyz.shape != (scores.shape[1], 3) or truth.shape != (85, 3):
        raise ValueError("diagnostic candidate/target shapes are inconsistent")
    errors = np.linalg.norm(truth[:, None, :] - xyz[None, :, :], axis=-1)
    errors *= float(scale_mm)
    sample_indices = np.arange(scores.shape[1])
    ranks = np.empty(85, dtype=np.int64)
    oracle = {int(k): np.empty(85, dtype=np.float32) for k in top_k}
    for landmark in range(85):
        order = np.lexsort((sample_indices, -scores[landmark]))
        nearest = int(np.argmin(errors[landmark]))
        ranks[landmark] = int(np.flatnonzero(order == nearest)[0]) + 1
        for k in oracle:
            oracle[k][landmark] = float(
                np.min(errors[landmark, order[: min(k, len(order))]])
            )
    return ranks, oracle


def confounding_probability_mass(
    logits: np.ndarray,
    candidates: np.ndarray,
    targets: np.ndarray,
    scale_mm: float,
    geodesic_distances_mm: np.ndarray,
    candidate_normals: np.ndarray,
    source_normals: np.ndarray,
    euclidean_close_mm: float,
    geodesic_far_mm: float,
    normal_dot_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    probabilities = torch.softmax(scores, dim=-1).numpy().astype(np.float64)
    euclidean = np.linalg.norm(
        np.asarray(targets)[:, None, :] - np.asarray(candidates)[None, :, :],
        axis=-1,
    ) * float(scale_mm)
    geodesic = np.asarray(geodesic_distances_mm, dtype=np.float64)
    shortcut_mask = (euclidean <= euclidean_close_mm) & (
        (~np.isfinite(geodesic)) | (geodesic >= geodesic_far_mm)
    )
    dots = np.einsum(
        "nd,ld->ln",
        np.asarray(candidate_normals, dtype=np.float64),
        np.asarray(source_normals, dtype=np.float64),
    )
    normal_mask = dots < float(normal_dot_threshold)
    return (
        np.sum(probabilities * shortcut_mask, axis=-1).astype(np.float32),
        np.sum(probabilities * normal_mask, axis=-1).astype(np.float32),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--geodesic-cache-dir", required=True)
    parser.add_argument("--top-k", nargs="+", type=_positive_int, default=[1, 8, 32, 64])
    parser.add_argument("--euclidean-close-mm", type=_positive_float, default=4.0)
    parser.add_argument("--geodesic-far-mm", type=_positive_float, default=8.0)
    parser.add_argument("--normal-dot-threshold", type=_bounded_dot, default=0.0)
    parser.add_argument("--run-seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    _, model, model_config, data_config = _load_model(args.checkpoint_path, device)
    if not isinstance(model, PointNeXtSurfaceHeatmapRegressor):
        raise ValueError(
            "geodesic diagnostics require an independent-ear PointNeXt heatmap checkpoint"
        )
    if str(model_config.get("decoder")) != "surface_heatmap":
        raise ValueError("checkpoint is not a surface-heatmap model")

    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    dataset_ids = [dataset.get_identifier(index) for index in range(len(dataset))]
    subject_index = {value: index for index, value in enumerate(dataset_ids)}
    validation_ids, run_seed, num_points = _validate_context(
        args, data_config, dataset
    )
    all_fold_ids = list(data_config["train_ids"]) + list(validation_ids)
    validate_geodesic_manifest(
        args.geodesic_cache_dir,
        int(data_config["outer_fold"]),
        all_fold_ids,
        args.folds_json,
        args.predictions_json,
        args.calibration_json,
    )
    top_k = sorted(set(int(value) for value in args.top_k))
    if top_k[-1] > num_points:
        raise ValueError("diagnostic top-k exceeds the checkpoint point count")
    if args.geodesic_far_mm <= args.euclidean_close_mm:
        raise ValueError("geodesic-far threshold must exceed Euclidean-close threshold")

    calibration = read_json(args.calibration_json)
    predictions = load_center_predictions(args.predictions_json, dataset_ids)
    ranks = []
    oracle = {k: [] for k in top_k}
    shortcut_mass = []
    normal_mass = []
    source_projection_errors = []
    per_ear = {}
    started = time.perf_counter()
    total = len(validation_ids) * 2
    complete = 0
    for subject_position, subject_id in enumerate(validation_ids):
        mesh, left, right = dataset[subject_index[subject_id]]
        for ear_offset, (ear, target_world) in enumerate(
            zip(EAR_NAMES, (left, right))
        ):
            item = subject_position * 2 + ear_offset
            prepared = prepare_ear_geometry(
                mesh,
                target_world,
                ear,
                predictions[prediction_key(subject_id, ear)],
                calibration,
                num_points,
                run_seed + 100_000 + item * 1009,
                include_sampling_metadata=True,
            )
            cache = load_geodesic_cache_entry(
                cache_path(args.geodesic_cache_dir, subject_id, ear),
                prepared.crop_mesh,
            )
            geodesic = sample_geodesic_distances_from_barycentric(
                prepared.crop_mesh,
                prepared.sample_face_indices,
                prepared.sample_barycentric,
                cache,
            )
            points = torch.from_numpy(prepared.point_features).unsqueeze(0).to(device)
            with torch.no_grad(), checkpoint_autocast_context(device, model_config):
                details = model.forward_with_details(points)
            logits = details["heatmap_logits"].squeeze(0).float().cpu().numpy()
            candidates = details["surface_candidates"].squeeze(0).float().cpu().numpy()
            ear_ranks, ear_oracle = candidate_retrieval_metrics(
                logits,
                candidates,
                prepared.target,
                float(calibration["local_scale"]),
                top_k,
            )
            ear_shortcut, ear_normal = confounding_probability_mass(
                logits,
                candidates,
                prepared.target,
                float(calibration["local_scale"]),
                geodesic,
                prepared.point_features[:, 3:6],
                cache["source_normals_canonical"],
                args.euclidean_close_mm,
                args.geodesic_far_mm,
                args.normal_dot_threshold,
            )
            ranks.append(ear_ranks)
            for k in top_k:
                oracle[k].append(ear_oracle[k])
            shortcut_mass.append(ear_shortcut)
            normal_mass.append(ear_normal)
            source_projection_errors.append(
                np.asarray(cache["source_projection_error_mm"], dtype=np.float32)
            )
            key = prediction_key(subject_id, ear)
            per_ear[key] = {
                "mean_nearest_correct_rank": float(np.mean(ear_ranks)),
                "recall_at": {
                    str(k): float(np.mean(ear_ranks <= k)) for k in top_k
                },
                "topk_oracle_md_mm": {
                    str(k): float(np.mean(ear_oracle[k])) for k in top_k
                },
                "euclidean_close_geodesic_far_mass": float(np.mean(ear_shortcut)),
                "strong_normal_difference_mass": float(np.mean(ear_normal)),
                "source_projection_md_mm": float(
                    np.mean(cache["source_projection_error_mm"])
                ),
            }
            complete += 1
            print(f"Analyzed geodesic candidates {complete}/{total}: {key}")

    rank_values = np.stack(ranks)
    shortcut_values = np.stack(shortcut_mass)
    normal_values = np.stack(normal_mass)
    report = {
        "schema_version": 1,
        "component": "fold_geodesic_heatmap_candidate_diagnostic",
        "outer_fold": int(data_config["outer_fold"]),
        "run_seed": int(run_seed),
        "subject_count": len(validation_ids),
        "ear_count": total,
        "num_points": int(num_points),
        "thresholds": {
            "euclidean_close_mm": float(args.euclidean_close_mm),
            "geodesic_far_mm": float(args.geodesic_far_mm),
            "strong_normal_dot_below": float(args.normal_dot_threshold),
        },
        "nearest_correct_candidate_rank": {
            "distribution": _distribution(rank_values),
            "recall_at": {
                str(k): float(np.mean(rank_values <= k)) for k in top_k
            },
            "per_landmark_mean": rank_values.mean(axis=0).tolist(),
        },
        "topk_candidate_oracle_mm": {
            str(k): {
                "distribution": _distribution(np.stack(oracle[k])),
                "pooled_md_mm": float(np.mean(oracle[k])),
                "per_landmark_md_mm": np.stack(oracle[k]).mean(axis=0).tolist(),
            }
            for k in top_k
        },
        "confounding_probability_mass": {
            "euclidean_close_geodesic_far": {
                "distribution": _distribution(shortcut_values),
                "per_landmark_mean": shortcut_values.mean(axis=0).tolist(),
            },
            "strong_normal_difference": {
                "distribution": _distribution(normal_values),
                "per_landmark_mean": normal_values.mean(axis=0).tolist(),
            },
        },
        "annotation_to_crop_surface_error_mm": {
            "distribution": _distribution(np.stack(source_projection_errors)),
            "pooled_md_mm": float(np.mean(source_projection_errors)),
            "per_landmark_md_mm": np.stack(source_projection_errors).mean(axis=0).tolist(),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "per_ear": per_ear,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    recall = report["nearest_correct_candidate_rank"]["recall_at"]
    print("Candidate recall: " + ", ".join(f"@{k}={recall[str(k)]:.4f}" for k in top_k))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
