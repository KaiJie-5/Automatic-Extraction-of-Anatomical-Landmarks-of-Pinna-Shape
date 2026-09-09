"""Screen an inference-only confidence gate between baseline and cascade models.

The gate thresholds are calibrated from inference signals on the outer-training
subjects without consulting their landmark errors.  The held-out fold is used
only to score the explicitly enumerated feature/scope/quantile/blend choices.
This preserves the existing baseline and cascade checkpoints and makes a later
confirmation run reproducible with one fixed gate configuration.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ..dataset import Dataset
from ..pipeline_dataset import EAR_NAMES, prediction_key, prepare_ear_geometry
from ..pointnet2_utils import index_points
from ..precision import checkpoint_autocast_context
from ..surface import project_points_to_mesh
from .evaluate_prior import (
    CONTOURS,
    _device,
    _distribution,
    _load_model,
    _part_means,
    _validate_context,
    _validate_prior_manifest,
    _world_from_local,
)
from .fitting import file_sha256, load_center_predictions, read_json
from .pca import PCAShapePrior


CASCADE_CONFIG_KEYS = {
    "cascade_stages",
    "cascade_attention_heads",
    "cascade_radius_normalized",
    "cascade_radius_decay",
    "cascade_dropout",
}
GATE_FEATURES = (
    "entropy",
    "inverse_peak_probability",
    "inverse_peak_margin",
    "spatial_spread_mm",
    "refinement_shift_mm",
    "pca_correction_mm",
    "model_disagreement_mm",
)
GATE_SCOPES = ("ear", "landmark")


def _bounded_open_unit(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("gate quantiles must be finite and in (0, 1)")
    return parsed


def _bounded_positive_unit(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("gate blends must be finite and in (0, 1]")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint-path", required=True)
    parser.add_argument("--cascade-checkpoint-path", required=True)
    parser.add_argument("--prior-path", required=True)
    parser.add_argument("--prior-manifest", required=True)
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--components", type=_positive_int, default=32)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--run-seed", type=int)
    parser.add_argument(
        "--gate-features",
        nargs="+",
        choices=GATE_FEATURES,
        default=list(GATE_FEATURES),
    )
    parser.add_argument(
        "--gate-scopes",
        nargs="+",
        choices=GATE_SCOPES,
        default=list(GATE_SCOPES),
    )
    parser.add_argument(
        "--gate-quantiles",
        nargs="+",
        type=_bounded_open_unit,
        default=[0.50, 0.65, 0.80, 0.90],
        help="outer-training uncertainty quantiles above which the cascade is used",
    )
    parser.add_argument(
        "--gate-blends",
        nargs="+",
        type=_bounded_positive_unit,
        default=[0.25, 0.50, 0.75, 1.0],
        help="fraction of the cascade correction applied to selected outputs",
    )
    parser.add_argument(
        "--projection-shortlist",
        type=_positive_int,
        default=24,
        help="number of lowest-PCA-MD gates receiving exact triangle projection",
    )
    parser.add_argument(
        "--projection-workers",
        type=_positive_int,
        default=10,
        help="CPU threads used for exact point-to-triangle projection",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def _base_model_config(config: Mapping[str, object]) -> dict:
    return {
        key: value
        for key, value in dict(config).items()
        if key not in CASCADE_CONFIG_KEYS
    }


def _validate_model_pair(
    baseline_checkpoint: Mapping[str, object],
    baseline_config: Mapping[str, object],
    baseline_data: Mapping[str, object],
    cascade_checkpoint: Mapping[str, object],
    cascade_config: Mapping[str, object],
    cascade_data: Mapping[str, object],
    baseline_checkpoint_path: str | Path,
) -> None:
    for label, config in (("baseline", baseline_config), ("cascade", cascade_config)):
        if (
            str(config.get("backbone")) != "pointnext"
            or str(config.get("decoder")) != "surface_heatmap"
            or str(config.get("bilateral_mode", "none")) != "none"
        ):
            raise ValueError(
                f"{label} checkpoint must be a single-ear PointNeXt surface-heatmap model"
            )
    if int(baseline_config.get("cascade_stages", 0)) != 0:
        raise ValueError("baseline checkpoint must have cascade_stages=0")
    if int(cascade_config.get("cascade_stages", 0)) <= 0:
        raise ValueError("cascade checkpoint must contain at least one cascade stage")
    if _base_model_config(baseline_config) != _base_model_config(cascade_config):
        raise ValueError(
            "baseline and cascade model configurations differ outside cascade settings"
        )
    for key in (
        "outer_fold",
        "train_ids",
        "validation_ids",
        "num_points",
        "calibration",
        "artifact_checksums",
        "seed",
    ):
        if baseline_data.get(key) != cascade_data.get(key):
            raise ValueError(f"baseline and cascade checkpoint {key} values differ")
    initialization = cascade_data.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("cascade checkpoint does not record its baseline initialization")
    expected_hash = file_sha256(baseline_checkpoint_path)
    if initialization.get("source_checkpoint_sha256") != expected_hash:
        raise ValueError(
            "cascade checkpoint was not initialized from the supplied baseline checkpoint"
        )
    if "model_state_dict" not in baseline_checkpoint or "model_state_dict" not in cascade_checkpoint:
        raise ValueError("both checkpoints must contain model_state_dict")


def _surface_spread_mm(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    topk: int,
    temperature: float,
    local_scale: float,
) -> torch.Tensor:
    count = min(int(topk), int(logits.shape[-1]))
    values, indices = logits.float().topk(count, dim=-1, sorted=True)
    selected = index_points(candidates.float(), indices)
    weights = torch.softmax(values / float(temperature), dim=-1)
    center = (selected * weights.unsqueeze(-1)).sum(dim=2)
    squared = (selected - center.unsqueeze(2)).square().sum(dim=-1)
    return torch.sqrt((weights * squared).sum(dim=-1).clamp_min(0.0)) * float(
        local_scale
    )


def _heatmap_uncertainty(logits: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Compute inference-only entropy and peak statistics per landmark."""
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    probabilities = torch.exp(log_probabilities)
    denominator = max(math.log(max(int(logits.shape[-1]), 2)), 1.0)
    entropy = -(probabilities * log_probabilities).sum(dim=-1) / denominator
    top = probabilities.topk(min(2, int(probabilities.shape[-1])), dim=-1).values
    peak = top[..., 0]
    margin = peak - top[..., 1] if top.shape[-1] == 2 else peak
    return {
        "entropy": entropy,
        "peak_probability": peak,
        "peak_margin": margin,
    }


def _finite_landmark_values(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (85,) or not np.isfinite(array).all():
        raise RuntimeError(f"{name} must contain 85 finite landmark values")
    return array


def _predict_pair(
    baseline_model,
    baseline_config: Mapping[str, object],
    cascade_model,
    cascade_config: Mapping[str, object],
    prepared,
    prior: PCAShapePrior,
    components: int,
    beta: float,
    local_scale: float,
    device: torch.device,
) -> dict:
    values = torch.from_numpy(
        prepared.point_features.astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad(), checkpoint_autocast_context(device, baseline_config):
        baseline = baseline_model.forward_with_details(values)
    with torch.no_grad(), checkpoint_autocast_context(device, cascade_config):
        cascade = cascade_model.forward_with_details(values)
    required = {
        "final",
        "coarse",
        "heatmap_logits",
        "surface_candidates",
    }
    for label, details in (("baseline", baseline), ("cascade", cascade)):
        missing = sorted(required.difference(details))
        if missing:
            raise ValueError(f"{label} model details are missing {missing}")

    baseline_final = baseline["final"].squeeze(0).float().cpu().numpy()
    cascade_final = cascade["final"].squeeze(0).float().cpu().numpy()
    if (
        baseline_final.shape != (85, 3)
        or cascade_final.shape != (85, 3)
        or not np.isfinite(baseline_final).all()
        or not np.isfinite(cascade_final).all()
    ):
        raise RuntimeError("model pair did not produce finite (85, 3) predictions")

    logits = baseline["heatmap_logits"].float()
    uncertainty = _heatmap_uncertainty(logits)
    spread = _surface_spread_mm(
        logits,
        baseline["surface_candidates"],
        int(baseline_model.heatmap_topk),
        float(baseline_model.heatmap_coordinate_temperature),
        local_scale,
    )
    coarse = baseline["coarse"].squeeze(0).float().cpu().numpy()
    baseline_pca = prior.blend(
        baseline_final,
        beta=beta,
        n_components=components,
    )
    features = {
        "entropy": uncertainty["entropy"].squeeze(0).cpu().numpy(),
        "inverse_peak_probability": -uncertainty["peak_probability"]
        .squeeze(0)
        .cpu()
        .numpy(),
        "inverse_peak_margin": -uncertainty["peak_margin"]
        .squeeze(0)
        .cpu()
        .numpy(),
        "spatial_spread_mm": spread.squeeze(0).cpu().numpy(),
        "refinement_shift_mm": np.linalg.norm(
            baseline_final - coarse, axis=1
        )
        * local_scale,
        "pca_correction_mm": np.linalg.norm(
            baseline_pca - baseline_final, axis=1
        )
        * local_scale,
        "model_disagreement_mm": np.linalg.norm(
            cascade_final - baseline_final, axis=1
        )
        * local_scale,
    }
    features = {
        name: _finite_landmark_values(name, feature)
        for name, feature in features.items()
    }
    return {
        "baseline_local": baseline_final.astype(np.float32),
        "cascade_local": cascade_final.astype(np.float32),
        "features": features,
    }


def calibrate_gate_threshold(
    training_values: np.ndarray,
    scope: str,
    quantile: float,
) -> float | np.ndarray:
    values = np.asarray(training_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 85 or not np.isfinite(values).all():
        raise ValueError("training gate features must have shape (ears, 85)")
    if scope == "ear":
        return float(np.quantile(values.mean(axis=1), quantile))
    if scope == "landmark":
        return np.quantile(values, quantile, axis=0)
    raise ValueError(f"unknown gate scope: {scope}")


def gate_alpha(
    feature_values: np.ndarray,
    threshold: float | np.ndarray,
    scope: str,
    blend: float,
) -> np.ndarray:
    values = np.asarray(feature_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 85 or not np.isfinite(values).all():
        raise ValueError("gate features must have shape (ears, 85)")
    if not math.isfinite(float(blend)) or not 0.0 < float(blend) <= 1.0:
        raise ValueError("gate blend must be in (0, 1]")
    if scope == "ear":
        scalar = float(threshold)
        mask = values.mean(axis=1, keepdims=True) >= scalar
        mask = np.broadcast_to(mask, values.shape)
    elif scope == "landmark":
        vector = np.asarray(threshold, dtype=np.float64)
        if vector.shape != (85,) or not np.isfinite(vector).all():
            raise ValueError("landmark gate threshold must have shape (85,)")
        mask = values >= vector[None, :]
    else:
        raise ValueError(f"unknown gate scope: {scope}")
    return mask.astype(np.float32) * float(blend)


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    if (
        x.shape != y.shape
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or float(np.std(x)) <= 1e-12
        or float(np.std(y)) <= 1e-12
    ):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _errors_local(
    prediction: np.ndarray,
    target: np.ndarray,
    local_scale: float,
) -> np.ndarray:
    errors = np.linalg.norm(
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(target, dtype=np.float64),
        axis=1,
    ) * float(local_scale)
    return _finite_landmark_values("local landmark errors", errors).astype(np.float32)


def _metric_block(errors: np.ndarray) -> dict:
    values = np.asarray(errors, dtype=np.float64)
    ear_md = values.mean(axis=1)
    return {
        "pooled_md_mm": float(values.mean()),
        "per_part_md_mm": _part_means(values),
        "ear_distribution_mm": _distribution(ear_md),
    }


def _threshold_json(threshold: float | np.ndarray) -> float | list[float]:
    if np.isscalar(threshold):
        return float(threshold)
    return np.asarray(threshold, dtype=np.float64).tolist()


def _configuration_id(
    feature: str,
    scope: str,
    quantile: float,
    blend: float,
) -> str:
    return (
        f"{feature}__{scope}__q{quantile:.3f}__blend{blend:.3f}"
        .replace(".", "p")
    )


def _collect_records(
    subject_ids: Sequence[str],
    seed_offset: int,
    label: str,
    dataset: Dataset,
    subject_index: Mapping[str, int],
    predictions: Mapping[str, np.ndarray],
    calibration: Mapping[str, object],
    num_points: int,
    run_seed: int,
    baseline_model,
    baseline_config: Mapping[str, object],
    cascade_model,
    cascade_config: Mapping[str, object],
    prior: PCAShapePrior,
    components: int,
    beta: float,
    device: torch.device,
    retain_geometry: bool,
) -> list[dict]:
    records = []
    total = len(subject_ids) * len(EAR_NAMES)
    completed = 0
    local_scale = float(calibration["local_scale"])
    for subject_position, subject_id in enumerate(subject_ids):
        mesh, left, right = dataset[subject_index[subject_id]]
        ground_truth_by_ear = {"left": left, "right": right}
        for ear_offset, ear in enumerate(EAR_NAMES):
            item = subject_position * len(EAR_NAMES) + ear_offset
            ground_truth = ground_truth_by_ear[ear]
            prepared = prepare_ear_geometry(
                mesh,
                ground_truth,
                ear,
                predictions[prediction_key(subject_id, ear)],
                calibration,
                num_points,
                run_seed + seed_offset + item * 1009,
            )
            record = _predict_pair(
                baseline_model,
                baseline_config,
                cascade_model,
                cascade_config,
                prepared,
                prior,
                components,
                beta,
                local_scale,
                device,
            )
            record.update(
                {
                    "subject_id": subject_id,
                    "ear": ear,
                }
            )
            if retain_geometry:
                record["target_local"] = prepared.target.astype(np.float32)
                record["ground_truth_world"] = np.asarray(
                    ground_truth, dtype=np.float32
                )
                record["prepared"] = prepared
            records.append(record)
            completed += 1
            print(f"Collected {label} gate signals {completed}/{total}: {subject_id}:{ear}")
    return records


def _evaluate_configuration(
    records: Sequence[Mapping[str, object]],
    alpha: np.ndarray,
    prior: PCAShapePrior,
    components: int,
    beta: float,
    local_scale: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    raw_predictions = []
    pca_predictions = []
    raw_errors = []
    pca_errors = []
    for index, record in enumerate(records):
        baseline = np.asarray(record["baseline_local"], dtype=np.float32)
        cascade = np.asarray(record["cascade_local"], dtype=np.float32)
        mixed = baseline + alpha[index, :, None] * (cascade - baseline)
        pca = prior.blend(mixed, beta=beta, n_components=components).astype(np.float32)
        raw_predictions.append(mixed.astype(np.float32))
        pca_predictions.append(pca)
        raw_errors.append(
            _errors_local(mixed, record["target_local"], local_scale)
        )
        pca_errors.append(
            _errors_local(pca, record["target_local"], local_scale)
        )
    raw_stack = np.stack(raw_errors)
    pca_stack = np.stack(pca_errors)
    metrics = {
        "selected_landmark_fraction": float(np.mean(alpha > 0.0)),
        "mean_gate_alpha": float(np.mean(alpha)),
        "raw": _metric_block(raw_stack),
        "pca": _metric_block(pca_stack),
    }
    return metrics, np.stack(raw_predictions), np.stack(pca_predictions)


def _project_configuration(
    records: Sequence[Mapping[str, object]],
    raw_predictions: np.ndarray,
    pca_predictions: np.ndarray,
    executor: ThreadPoolExecutor,
) -> tuple[dict, np.ndarray]:
    def project_one(values):
        record, raw_local, pca_local = values
        prepared = record["prepared"]
        ear = str(record["ear"])
        ground_truth = np.asarray(record["ground_truth_world"], dtype=np.float32)
        raw_world = _world_from_local(prepared, ear, raw_local)
        pca_world = _world_from_local(prepared, ear, pca_local)
        projected = np.linalg.norm(
            project_points_to_mesh(raw_world, prepared.crop_mesh) - ground_truth,
            axis=1,
        ).astype(np.float32)
        pca_projected = np.linalg.norm(
            project_points_to_mesh(pca_world, prepared.crop_mesh) - ground_truth,
            axis=1,
        ).astype(np.float32)
        return projected, pca_projected

    rows = executor.map(
        project_one,
        zip(records, raw_predictions, pca_predictions),
    )
    projected_errors = []
    pca_projected_errors = []
    for projected, pca_projected in rows:
        projected_errors.append(projected)
        pca_projected_errors.append(pca_projected)
    projected_stack = np.stack(projected_errors)
    pca_projected_stack = np.stack(pca_projected_errors)
    return {
        "projected": _metric_block(projected_stack),
        "pca_projected": _metric_block(pca_projected_stack),
    }, pca_projected_stack


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not math.isfinite(float(args.beta)) or not 0.0 <= float(args.beta) <= 1.0:
        raise ValueError("--beta must be finite and in [0, 1]")
    if len(set(args.gate_features)) != len(args.gate_features):
        raise ValueError("--gate-features contains duplicates")
    if len(set(args.gate_scopes)) != len(args.gate_scopes):
        raise ValueError("--gate-scopes contains duplicates")

    device = _device(args.device)
    baseline_checkpoint, baseline_model, baseline_config, baseline_data = _load_model(
        args.baseline_checkpoint_path, device
    )
    cascade_checkpoint, cascade_model, cascade_config, cascade_data = _load_model(
        args.cascade_checkpoint_path, device
    )
    _validate_model_pair(
        baseline_checkpoint,
        baseline_config,
        baseline_data,
        cascade_checkpoint,
        cascade_config,
        cascade_data,
        args.baseline_checkpoint_path,
    )
    # The reconstructed models own their parameters; checkpoint tensor copies
    # are no longer needed and otherwise double peak GPU memory.
    del baseline_checkpoint, cascade_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    all_subject_ids = [
        dataset.get_identifier(index) for index in range(len(dataset))
    ]
    subject_index = {
        identifier: index for index, identifier in enumerate(all_subject_ids)
    }
    validation_ids, run_seed, num_points = _validate_context(
        args, baseline_data, dataset
    )
    candidate_validation_ids, candidate_seed, candidate_num_points = _validate_context(
        args, cascade_data, dataset
    )
    if (
        validation_ids != candidate_validation_ids
        or run_seed != candidate_seed
        or num_points != candidate_num_points
    ):
        raise ValueError("baseline and cascade validation contexts differ")

    calibration = read_json(args.calibration_json)
    predictions = load_center_predictions(args.predictions_json, all_subject_ids)
    prior = PCAShapePrior.load(args.prior_path)
    prior_manifest = read_json(args.prior_manifest)
    _validate_prior_manifest(
        prior_manifest,
        args,
        int(baseline_data["outer_fold"]),
        list(baseline_data["train_ids"]),
        prior,
    )
    components = int(args.components)
    if components > int(prior.n_components):
        raise ValueError(
            f"requested {components} PCA components but prior stores {prior.n_components}"
        )
    beta = float(args.beta)
    train_ids = list(baseline_data["train_ids"])

    training_records = _collect_records(
        train_ids,
        200_000,
        "outer-training",
        dataset,
        subject_index,
        predictions,
        calibration,
        num_points,
        run_seed,
        baseline_model,
        baseline_config,
        cascade_model,
        cascade_config,
        prior,
        components,
        beta,
        device,
        retain_geometry=False,
    )
    validation_records = _collect_records(
        validation_ids,
        100_000,
        "held-out",
        dataset,
        subject_index,
        predictions,
        calibration,
        num_points,
        run_seed,
        baseline_model,
        baseline_config,
        cascade_model,
        cascade_config,
        prior,
        components,
        beta,
        device,
        retain_geometry=True,
    )
    local_scale = float(calibration["local_scale"])
    train_features = {
        feature: np.stack(
            [record["features"][feature] for record in training_records]
        )
        for feature in args.gate_features
    }
    validation_features = {
        feature: np.stack(
            [record["features"][feature] for record in validation_records]
        )
        for feature in args.gate_features
    }

    ear_count = len(validation_records)
    configurations = {}
    prediction_cache = {}
    endpoint_alpha = {
        "baseline": np.zeros((ear_count, 85), dtype=np.float32),
        "cascade": np.ones((ear_count, 85), dtype=np.float32),
    }
    for name, alpha in endpoint_alpha.items():
        metrics, raw_predictions, pca_predictions = _evaluate_configuration(
            validation_records,
            alpha,
            prior,
            components,
            beta,
            local_scale,
        )
        configurations[name] = {
            "kind": "endpoint",
            **metrics,
        }
        prediction_cache[name] = (raw_predictions, pca_predictions)

    for feature in args.gate_features:
        for scope in args.gate_scopes:
            for quantile in sorted(set(float(value) for value in args.gate_quantiles)):
                threshold = calibrate_gate_threshold(
                    train_features[feature], scope, quantile
                )
                for blend in sorted(set(float(value) for value in args.gate_blends)):
                    name = _configuration_id(feature, scope, quantile, blend)
                    alpha = gate_alpha(
                        validation_features[feature], threshold, scope, blend
                    )
                    metrics, raw_predictions, pca_predictions = _evaluate_configuration(
                        validation_records,
                        alpha,
                        prior,
                        components,
                        beta,
                        local_scale,
                    )
                    configurations[name] = {
                        "kind": "confidence_gate",
                        "feature": feature,
                        "scope": scope,
                        "training_quantile": quantile,
                        "training_threshold": _threshold_json(threshold),
                        "blend": blend,
                        **metrics,
                    }
                    prediction_cache[name] = (raw_predictions, pca_predictions)

    grid_names = [
        name for name, item in configurations.items() if item["kind"] == "confidence_gate"
    ]
    shortlist = sorted(
        grid_names,
        key=lambda name: (
            configurations[name]["pca"]["pooled_md_mm"],
            configurations[name]["pca"]["ear_distribution_mm"]["p95"],
            name,
        ),
    )[: int(args.projection_shortlist)]
    projected_names = ["baseline", "cascade", *shortlist]
    projected_error_cache = {}
    with ThreadPoolExecutor(max_workers=int(args.projection_workers)) as executor:
        for index, name in enumerate(projected_names, start=1):
            raw_predictions, pca_predictions = prediction_cache[name]
            projection_metrics, pca_projected_errors = _project_configuration(
                validation_records, raw_predictions, pca_predictions, executor
            )
            configurations[name].update(projection_metrics)
            projected_error_cache[name] = pca_projected_errors
            print(
                f"Projected gate configuration {index}/{len(projected_names)}: {name}"
            )

    selected_name = min(
        projected_names,
        key=lambda name: (
            configurations[name]["pca_projected"]["pooled_md_mm"],
            configurations[name]["pca_projected"]["ear_distribution_mm"]["p95"],
            name,
        ),
    )
    baseline_projected = projected_error_cache["baseline"]
    cascade_projected = projected_error_cache["cascade"]
    cascade_gain = baseline_projected - cascade_projected
    baseline_error = baseline_projected
    diagnostics = {}
    for feature in args.gate_features:
        values = validation_features[feature]
        diagnostics[feature] = {
            "landmark_feature_vs_baseline_error": _pearson(values, baseline_error),
            "landmark_feature_vs_cascade_gain": _pearson(values, cascade_gain),
            "ear_mean_feature_vs_baseline_error": _pearson(
                values.mean(axis=1), baseline_error.mean(axis=1)
            ),
            "ear_mean_feature_vs_cascade_gain": _pearson(
                values.mean(axis=1), cascade_gain.mean(axis=1)
            ),
        }

    selected_errors = projected_error_cache[selected_name]
    baseline_ear = baseline_projected.mean(axis=1)
    selected_ear = selected_errors.mean(axis=1)
    per_ear = {}
    for index, record in enumerate(validation_records):
        key = prediction_key(str(record["subject_id"]), str(record["ear"]))
        per_ear[key] = {
            "baseline_pca_projected_md_mm": float(baseline_ear[index]),
            "cascade_pca_projected_md_mm": float(
                cascade_projected[index].mean()
            ),
            "selected_pca_projected_md_mm": float(selected_ear[index]),
            "selected_improvement_mm": float(baseline_ear[index] - selected_ear[index]),
        }

    selection_mode = (
        "fixed_confirmation"
        if len(grid_names) == 1
        else "fold0_grid_screen"
    )
    report = {
        "schema_version": 1,
        "component": "fold_confidence_gated_landmark_cascade_evaluation",
        "selection_mode": selection_mode,
        "warning": (
            "When selection_mode is fold0_grid_screen, selected is a screening result "
            "and must be fixed before multi-fold/multi-seed confirmation."
        ),
        "outer_fold": int(baseline_data["outer_fold"]),
        "run_seed": int(run_seed),
        "subject_count": len(validation_ids),
        "ear_count": ear_count,
        "training_calibration_subject_count": len(train_ids),
        "training_calibration_ear_count": len(training_records),
        "training_calibration": {
            "uses_landmark_error_labels": False,
            "scope": "outer-training inference-signal distributions",
            "sample_seed_offset": 200_000,
        },
        "validation_sample_seed_offset": 100_000,
        "baseline_checkpoint_path": str(args.baseline_checkpoint_path),
        "baseline_checkpoint_sha256": file_sha256(args.baseline_checkpoint_path),
        "cascade_checkpoint_path": str(args.cascade_checkpoint_path),
        "cascade_checkpoint_sha256": file_sha256(args.cascade_checkpoint_path),
        "prior_path": str(args.prior_path),
        "prior_sha256": file_sha256(args.prior_path),
        "prior_manifest_path": str(args.prior_manifest),
        "components": components,
        "beta": beta,
        "gate_features": list(args.gate_features),
        "gate_scopes": list(args.gate_scopes),
        "gate_quantiles": sorted(set(float(value) for value in args.gate_quantiles)),
        "gate_blends": sorted(set(float(value) for value in args.gate_blends)),
        "projection_shortlist": int(args.projection_shortlist),
        "projection_workers": int(args.projection_workers),
        "configuration_count": len(configurations),
        "projected_configuration_count": len(projected_names),
        "uncertainty_diagnostics": diagnostics,
        "configurations": configurations,
        "selected": {"name": selected_name, **configurations[selected_name]},
        "selected_vs_baseline": {
            "mean_improvement_mm": float(baseline_ear.mean() - selected_ear.mean()),
            "ears_improved": int(np.sum(selected_ear < baseline_ear)),
            "ears_unchanged": int(np.sum(selected_ear == baseline_ear)),
            "ears_worsened": int(np.sum(selected_ear > baseline_ear)),
        },
        "per_ear": per_ear,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"Selected {selected_name}: "
        f"{configurations[selected_name]['pca_projected']['pooled_md_mm']:.6f} mm; "
        f"baseline {configurations['baseline']['pca_projected']['pooled_md_mm']:.6f} mm; "
        f"improvement {report['selected_vs_baseline']['mean_improvement_mm']:.6f} mm"
    )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
