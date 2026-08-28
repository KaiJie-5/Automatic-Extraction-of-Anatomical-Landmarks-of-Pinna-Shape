"""Qualitative prediction and importance viewer for proposal fold checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from src.canonical import decanonicalize_xyz, ear_bbox_center
from src.dataset import Dataset as MeshLandmarkDataset
from src.losses import ANCHOR_INDICES
from src.meshnet import meshnet_inputs_with_mesh
from src.pipeline_dataset import prediction_key, prepare_ear_geometry
from src.proposal_models import build_fold_landmark_model
from src.surface import project_points_to_mesh


PROJECT_ROOT = Path(
    "/iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape"
)
DATA_ROOT = PROJECT_ROOT / "data"
EAR_NAMES = ("left", "right")
CONTOURS = (
    ("outer", 0, 25),
    ("concha", 25, 55),
    ("inner", 55, 75),
    ("superior", 75, 85),
)
CONTOUR_BY_INDEX = tuple(
    name for name, start, end in CONTOURS for _ in range(start, end)
)

GREY = np.array([175, 180, 185, 130], dtype=np.uint8)
GREEN = np.array([36, 160, 88, 255], dtype=np.uint8)
ORANGE = np.array([240, 150, 45, 255], dtype=np.uint8)
RED = np.array([220, 55, 55, 255], dtype=np.uint8)
CYAN = np.array([25, 175, 205, 255], dtype=np.uint8)
PURPLE = np.array([135, 75, 185, 255], dtype=np.uint8)


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _clean_state_dict(state_dict):
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


@dataclass
class FoldContext:
    checkpoint_path: Path
    predictions_path: Path
    calibration_path: Path
    folds_path: Path
    checkpoint: Mapping[str, object]
    model: torch.nn.Module
    model_config: Mapping[str, object]
    data_config: Mapping[str, object]
    dataset: MeshLandmarkDataset
    subject_index: Mapping[str, int]
    predictions: Mapping[str, np.ndarray]
    calibration: Mapping[str, object]
    validation_ids: Sequence[str]
    outer_fold: int
    run_seed: int
    num_points: int
    backbone: str
    device: torch.device
    provenance_level: str


@dataclass
class EarTrace:
    subject_id: str
    ear: str
    sample_seed: int
    prepared: object
    input_features: np.ndarray
    neighbors: Optional[np.ndarray]
    input_xyz_world: np.ndarray
    target_local: np.ndarray
    ground_truth_world: np.ndarray
    coarse_local: np.ndarray
    final_local: np.ndarray
    coarse_world: np.ndarray
    final_world: np.ndarray
    projected_world: Optional[np.ndarray]
    simplified_mesh: Optional[trimesh.Trimesh]
    true_center_canonical: np.ndarray
    predicted_center_world: np.ndarray
    true_center_world: np.ndarray
    raw_errors_mm: np.ndarray
    coarse_errors_mm: np.ndarray
    projected_errors_mm: Optional[np.ndarray]


def _find_outer_fold(folds: Mapping[str, object], outer_fold: int):
    matches = [
        item for item in folds.get("outer", [])
        if int(item.get("fold", -1)) == outer_fold
    ]
    if len(matches) != 1:
        raise ValueError(f"folds JSON does not contain exactly one outer fold {outer_fold}")
    return matches[0]


def _validate_prediction_map(data, subject_ids: Sequence[str]) -> Mapping[str, np.ndarray]:
    if not isinstance(data, Mapping) or data.get("coordinate_frame") != "canonical_mm":
        raise ValueError("predictions JSON must declare coordinate_frame='canonical_mm'")
    raw = data.get("center_predictions")
    if not isinstance(raw, Mapping):
        raise ValueError("predictions JSON is missing center_predictions")
    expected = {
        prediction_key(subject, ear) for subject in subject_ids for ear in EAR_NAMES
    }
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "centre prediction coverage does not match the audited dataset: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    result = {}
    for key, value in raw.items():
        center = np.asarray(value, dtype=np.float32)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"centre prediction {key} is not a finite three-value vector")
        result[key] = center
    return result


def load_fold_context(args: argparse.Namespace) -> FoldContext:
    device = make_device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint = _torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "unsupported checkpoint: expected a proposal landmark component checkpoint, not a raw state dictionary"
        )
    if checkpoint.get("schema_version") == 2:
        raise ValueError("final v2 pipeline bundles are outside this fold-only visualizer's scope")
    if (
        checkpoint.get("component_schema_version") != 1
        or checkpoint.get("component") != "landmarks"
    ):
        raise ValueError(
            "unsupported checkpoint: expected component_schema_version=1 and component='landmarks'"
        )
    required = {"model_state_dict", "model_config", "data_config"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"incomplete proposal landmark checkpoint; missing: {missing}")

    model_config = dict(checkpoint["model_config"])
    data_config = dict(checkpoint["data_config"])
    backbone = str(model_config.get("backbone", ""))
    if backbone not in {"pointnet2", "pointnext", "pointtransformerv3", "meshnet"}:
        raise ValueError(f"unsupported proposal fold backbone: {backbone!r}")
    if backbone == "meshnet" and int(model_config.get("target_faces", 0)) <= 0:
        raise ValueError("MeshNet checkpoint is missing a positive target_faces value")
    outer_fold = int(data_config.get("outer_fold", -1))
    if outer_fold not in range(5):
        raise ValueError("checkpoint data_config is missing a valid outer_fold")

    dataset = MeshLandmarkDataset(args.mesh_dir, args.landmarks_dir)
    subject_ids = [dataset.get_identifier(index) for index in range(len(dataset))]
    subject_index = {subject: index for index, subject in enumerate(subject_ids)}
    folds_path = Path(args.folds_json)
    folds = _load_json(folds_path)
    checksum = hashlib.sha256(
        "\n".join(sorted(subject_ids)).encode("utf-8")
    ).hexdigest()
    if folds.get("subject_checksum") != checksum:
        raise ValueError("dataset subject IDs do not match the folds JSON checksum")
    fold = _find_outer_fold(folds, outer_fold)
    fold_train = list(fold.get("train", []))
    fold_validation = list(fold.get("validation", []))
    if (
        len(fold_train) != len(set(fold_train))
        or len(fold_validation) != len(set(fold_validation))
        or set(fold_train) & set(fold_validation)
        or set(fold_train) | set(fold_validation) != set(subject_ids)
    ):
        raise ValueError(
            "outer fold must cover every dataset subject exactly once without leakage"
        )
    if list(data_config.get("train_ids", [])) != list(fold.get("train", [])):
        raise ValueError("checkpoint training IDs do not exactly match folds JSON")
    if list(data_config.get("validation_ids", [])) != list(fold.get("validation", [])):
        raise ValueError("checkpoint validation IDs do not exactly match folds JSON")
    validation_ids = fold_validation

    calibration_path = Path(args.calibration_json)
    calibration = _load_json(calibration_path)
    if data_config.get("calibration") != calibration:
        raise ValueError("external calibration JSON does not exactly match checkpoint calibration")
    if int(calibration.get("schema_version", -1)) != 1:
        raise ValueError("unsupported crop calibration schema")
    local_scale = float(calibration.get("local_scale", 0.0))
    if not np.isfinite(local_scale) or local_scale <= 0:
        raise ValueError("calibration local_scale must be finite and positive")

    predictions_path = Path(args.predictions_json)
    predictions = _validate_prediction_map(_load_json(predictions_path), subject_ids)
    saved_seed = data_config.get("seed")
    if saved_seed is None and args.run_seed is None:
        raise ValueError("existing checkpoint does not store its run seed; provide --run-seed")
    if (
        saved_seed is not None
        and args.run_seed is not None
        and int(saved_seed) != int(args.run_seed)
    ):
        raise ValueError(
            f"--run-seed {args.run_seed} does not match checkpoint seed {saved_seed}"
        )
    run_seed = int(saved_seed if saved_seed is not None else args.run_seed)
    num_points = int(data_config.get("num_points", 0))
    if num_points <= 0:
        raise ValueError("checkpoint data_config is missing a positive num_points")

    checksum_fields = data_config.get("artifact_checksums")
    provenance_level = "structural_existing_checkpoint_without_artifact_hashes"
    if checksum_fields is not None:
        expected_hashes = {
            "folds_json_sha256": file_sha256(folds_path),
            "predictions_json_sha256": file_sha256(predictions_path),
            "calibration_json_sha256": file_sha256(calibration_path),
        }
        if dict(checksum_fields) != expected_hashes:
            raise ValueError("one or more artifact SHA-256 values do not match the checkpoint")
        provenance_level = "sha256_verified"

    model = build_fold_landmark_model(model_config).to(device)
    model.load_state_dict(_clean_state_dict(checkpoint["model_state_dict"]))
    model.eval()
    return FoldContext(
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        calibration_path=calibration_path,
        folds_path=folds_path,
        checkpoint=checkpoint,
        model=model,
        model_config=model_config,
        data_config=data_config,
        dataset=dataset,
        subject_index=subject_index,
        predictions=predictions,
        calibration=calibration,
        validation_ids=validation_ids,
        outer_fold=outer_fold,
        run_seed=run_seed,
        num_points=num_points,
        backbone=backbone,
        device=device,
        provenance_level=provenance_level,
    )


def validation_sample_seed(context: FoldContext, subject_id: str, ear: str) -> int:
    if subject_id not in context.validation_ids:
        raise ValueError(
            f"{subject_id} is not held out in outer fold {context.outer_fold}; training subjects are rejected"
        )
    if ear not in EAR_NAMES:
        raise ValueError(f"unsupported ear: {ear}")
    item = context.validation_ids.index(subject_id) * 2 + EAR_NAMES.index(ear)
    return int(context.run_seed + 100_000 + item * 1009)


def _model_details(
    context: FoldContext,
    values: torch.Tensor,
    neighbors: Optional[torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if context.backbone == "meshnet":
        details = context.model.forward_with_details(values, neighbors)
    else:
        details = context.model.forward_with_details(values)
    for name in ("coarse", "final"):
        if name not in details or tuple(details[name].shape[1:]) != (85, 3):
            raise RuntimeError(f"model {name} output is not shaped (B, 85, 3)")
        if not torch.isfinite(details[name]).all():
            raise RuntimeError(f"model {name} output contains non-finite values")
    return details


def prepare_trace(
    context: FoldContext,
    subject_id: str,
    ear: str,
    include_projection: bool = True,
) -> EarTrace:
    sample_seed = validation_sample_seed(context, subject_id, ear)
    mesh, left, right = context.dataset[context.subject_index[subject_id]]
    ground_truth = left if ear == "left" else right
    center = context.predictions[prediction_key(subject_id, ear)]
    sample_count = 1 if context.backbone == "meshnet" else context.num_points
    prepared = prepare_ear_geometry(
        mesh,
        ground_truth,
        ear,
        center,
        context.calibration,
        sample_count,
        sample_seed,
    )
    simplified_mesh = None
    neighbors_np = None
    if context.backbone == "meshnet":
        features, neighbors_np, simplified_mesh = meshnet_inputs_with_mesh(
            prepared.crop_mesh,
            int(context.model_config["target_faces"]),
            ear,
            prepared.transform,
        )
        input_features = features.astype(np.float32)
        input_xyz_canonical = prepared.transform.denormalize_xyz(input_features[:, :3])
        input_tensor = torch.from_numpy(input_features).unsqueeze(0).to(context.device)
        neighbors_tensor = torch.from_numpy(neighbors_np).unsqueeze(0).to(context.device)
    else:
        input_features = prepared.point_features.astype(np.float32)
        input_xyz_canonical = prepared.sampled_canonical_features[:, :3]
        input_tensor = torch.from_numpy(input_features).unsqueeze(0).to(context.device)
        neighbors_tensor = None

    with torch.no_grad():
        details = _model_details(context, input_tensor, neighbors_tensor)
    coarse_local = details["coarse"].squeeze(0).float().cpu().numpy().astype(np.float32)
    final_local = details["final"].squeeze(0).float().cpu().numpy().astype(np.float32)
    coarse_world = decanonicalize_xyz(
        prepared.transform.denormalize_xyz(coarse_local), ear
    ).astype(np.float32)
    final_world = decanonicalize_xyz(
        prepared.transform.denormalize_xyz(final_local), ear
    ).astype(np.float32)
    input_xyz_world = decanonicalize_xyz(input_xyz_canonical, ear).astype(np.float32)
    projected = (
        project_points_to_mesh(final_world, prepared.crop_mesh)
        if include_projection
        else None
    )
    true_center_canonical = ear_bbox_center(prepared.canonical_landmarks)
    predicted_center_world = decanonicalize_xyz(center, ear).astype(np.float32)
    true_center_world = decanonicalize_xyz(true_center_canonical, ear).astype(np.float32)
    return EarTrace(
        subject_id=subject_id,
        ear=ear,
        sample_seed=sample_seed,
        prepared=prepared,
        input_features=input_features,
        neighbors=neighbors_np,
        input_xyz_world=input_xyz_world,
        target_local=prepared.target,
        ground_truth_world=np.asarray(ground_truth, dtype=np.float32),
        coarse_local=coarse_local,
        final_local=final_local,
        coarse_world=coarse_world,
        final_world=final_world,
        projected_world=projected,
        simplified_mesh=simplified_mesh,
        true_center_canonical=true_center_canonical,
        predicted_center_world=predicted_center_world,
        true_center_world=true_center_world,
        raw_errors_mm=np.linalg.norm(final_world - ground_truth, axis=1).astype(np.float32),
        coarse_errors_mm=np.linalg.norm(coarse_world - ground_truth, axis=1).astype(np.float32),
        projected_errors_mm=(
            np.linalg.norm(projected - ground_truth, axis=1).astype(np.float32)
            if projected is not None
            else None
        ),
    )


def official_mean_distance_torch(
    prediction_local: torch.Tensor,
    target_local: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.linalg.norm(
        (prediction_local - target_local) * float(scale), dim=-1
    ).mean()


def gradient_importance(context: FoldContext, trace: EarTrace) -> Mapping[str, np.ndarray]:
    values = torch.from_numpy(trace.input_features).unsqueeze(0).to(context.device)
    values = values.clone().detach().requires_grad_(True)
    neighbors = (
        torch.from_numpy(trace.neighbors).unsqueeze(0).to(context.device)
        if trace.neighbors is not None
        else None
    )
    target = torch.from_numpy(trace.target_local).unsqueeze(0).to(context.device)
    context.model.zero_grad(set_to_none=True)
    details = _model_details(context, values, neighbors)
    score = official_mean_distance_torch(
        details["final"].float(), target.float(), trace.prepared.transform.scale
    )
    score.backward()
    gradient = values.grad.detach()[0].float().cpu().numpy()
    if context.backbone == "meshnet":
        xyz = np.linalg.norm(gradient[:, :12], axis=1)
        normals = np.linalg.norm(gradient[:, 12:15], axis=1)
    else:
        xyz = np.linalg.norm(gradient[:, :3], axis=1)
        normals = np.linalg.norm(gradient[:, 3:6], axis=1)
    return {
        "gradient_xyz": np.nan_to_num(
            xyz, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
        "gradient_normals": np.nan_to_num(
            normals, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
    }


def deterministic_spatial_clusters(
    xyz: np.ndarray, target_size: int = 128
) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("spatial clustering requires a nonempty (N, 3) array")
    if target_size <= 0:
        raise ValueError("occlusion cluster size must be positive")
    cluster_count = max(1, int(math.ceil(len(points) / int(target_size))))
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    anchors = [first]
    selected = np.zeros(len(points), dtype=bool)
    selected[first] = True
    nearest_sq = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, min(cluster_count, len(points))):
        scores = nearest_sq.copy()
        scores[selected] = -np.inf
        index = int(np.argmax(scores))
        anchors.append(index)
        selected[index] = True
        nearest_sq = np.minimum(
            nearest_sq, np.sum((points - points[index]) ** 2, axis=1)
        )
    anchor_points = points[np.asarray(anchors)]
    labels = np.empty(len(points), dtype=np.int32)
    for start in range(0, len(points), 4096):
        chunk = points[start : start + 4096]
        distances = np.sum(
            (chunk[:, None, :] - anchor_points[None, :, :]) ** 2, axis=2
        )
        labels[start : start + len(chunk)] = np.argmin(
            distances, axis=1
        ).astype(np.int32)
    return labels


def occlusion_importance(
    context: FoldContext,
    trace: EarTrace,
    cluster_size: int,
    batch_size: int,
) -> Mapping[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("occlusion batch size must be positive")
    labels = deterministic_spatial_clusters(trace.input_features[:, :3], cluster_size)
    group_ids = np.unique(labels)
    values = torch.from_numpy(trace.input_features).unsqueeze(0).to(context.device)
    target = torch.from_numpy(trace.target_local).unsqueeze(0).to(context.device)
    neighbors = (
        torch.from_numpy(trace.neighbors).unsqueeze(0).to(context.device)
        if trace.neighbors is not None
        else None
    )
    with torch.no_grad():
        baseline = _model_details(context, values, neighbors)["final"].float()
        baseline_md = official_mean_distance_torch(
            baseline, target.float(), trace.prepared.transform.scale
        )
    feature_mean = values.mean(dim=1, keepdim=True)
    delta_by_group = {}
    displacement_by_group = {}
    for start in range(0, len(group_ids), int(batch_size)):
        batch_groups = group_ids[start : start + int(batch_size)]
        occluded = values.repeat(len(batch_groups), 1, 1)
        for row, group in enumerate(batch_groups):
            mask = torch.from_numpy(labels == group).to(context.device)
            occluded[row, mask, :] = feature_mean[0, 0]
        batch_neighbors = (
            neighbors.repeat(len(batch_groups), 1, 1)
            if neighbors is not None
            else None
        )
        with torch.no_grad():
            predictions = _model_details(
                context, occluded, batch_neighbors
            )["final"].float()
            targets = target.repeat(len(batch_groups), 1, 1).float()
            errors = torch.linalg.norm(
                (predictions - targets) * trace.prepared.transform.scale,
                dim=-1,
            ).mean(dim=1)
            displacements = torch.linalg.norm(
                (predictions - baseline.repeat(len(batch_groups), 1, 1))
                * trace.prepared.transform.scale,
                dim=-1,
            ).mean(dim=1)
        for group, delta, displacement in zip(
            batch_groups, errors - baseline_md, displacements
        ):
            delta_by_group[int(group)] = float(delta.cpu())
            displacement_by_group[int(group)] = float(displacement.cpu())
    delta = np.asarray(
        [delta_by_group[int(group)] for group in labels], dtype=np.float32
    )
    displacement = np.asarray(
        [displacement_by_group[int(group)] for group in labels], dtype=np.float32
    )
    return {
        "cluster_labels": labels,
        "occlusion_md": delta,
        "occlusion_displacement": displacement,
    }


def normalize_importance(importance: np.ndarray) -> np.ndarray:
    importance = np.asarray(importance, dtype=np.float32)
    importance = np.nan_to_num(
        importance, nan=0.0, posinf=0.0, neginf=0.0
    )
    if importance.size == 0:
        return importance
    minimum = float(importance.min())
    maximum = float(importance.max())
    if maximum <= minimum:
        return np.zeros_like(importance, dtype=np.float32)
    return ((importance - minimum) / (maximum - minimum)).astype(np.float32)


def colorize_importance(importance: np.ndarray) -> np.ndarray:
    """Blue-to-cyan-to-yellow-to-red sequential RGBA heatmap."""
    values = normalize_importance(importance)
    colors = np.zeros((len(values), 4), dtype=np.uint8)
    colors[:, 3] = 255
    low = values < 0.33
    middle = (values >= 0.33) & (values < 0.66)
    high = values >= 0.66
    colors[low, 1] = np.clip(values[low] / 0.33 * 255, 0, 255).astype(np.uint8)
    colors[low, 2] = 255
    colors[middle, 0] = np.clip(
        (values[middle] - 0.33) / 0.33 * 255, 0, 255
    ).astype(np.uint8)
    colors[middle, 1] = 255
    colors[middle, 2] = np.clip(
        (1.0 - (values[middle] - 0.33) / 0.33) * 255, 0, 255
    ).astype(np.uint8)
    colors[high, 0] = 255
    colors[high, 1] = np.clip(
        (1.0 - (values[high] - 0.66) / 0.34) * 255, 0, 255
    ).astype(np.uint8)
    return colors


def colorize_signed(values: np.ndarray) -> np.ndarray:
    raw = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    maximum = float(np.max(np.abs(raw))) if raw.size else 0.0
    normalized = raw / maximum if maximum > 0 else np.zeros_like(raw)
    colors = np.empty((len(raw), 4), dtype=np.uint8)
    colors[:, 3] = 255
    positive = normalized >= 0
    strength = np.abs(normalized)
    colors[:, :3] = 220
    colors[positive, 0] = 230
    colors[positive, 1] = (220 * (1.0 - strength[positive])).astype(np.uint8)
    colors[positive, 2] = (220 * (1.0 - strength[positive])).astype(np.uint8)
    colors[~positive, 0] = (
        220 * (1.0 - strength[~positive])
    ).astype(np.uint8)
    colors[~positive, 1] = (
        220 * (1.0 - strength[~positive])
    ).astype(np.uint8)
    colors[~positive, 2] = 230
    return colors


def _set_equal_axes(axis, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.55, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def _box_edges(box) -> list[tuple[np.ndarray, np.ndarray]]:
    minimum, maximum = box.minimum, box.maximum
    corners = np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )
    edges = []
    for first in range(8):
        for second in range(first + 1, 8):
            if np.sum(corners[first] != corners[second]) == 1:
                edges.append((corners[first], corners[second]))
    return edges


def _plot_mesh(
    axis,
    mesh: trimesh.Trimesh,
    face_colors=None,
    alpha: float = 0.18,
) -> None:
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)
    if len(faces) > 20000:
        indices = np.linspace(0, len(faces) - 1, 20000, dtype=np.int64)
        faces = faces[indices]
        if face_colors is not None:
            face_colors = np.asarray(face_colors)[indices]
    collection = Poly3DCollection(vertices[faces], linewidths=0.0, alpha=alpha)
    if face_colors is None:
        collection.set_facecolor((0.68, 0.70, 0.72, alpha))
    else:
        collection.set_facecolor(np.asarray(face_colors, dtype=np.float32) / 255.0)
        collection.set_alpha(1.0)
    axis.add_collection3d(collection)


def _plot_polyline(axis, values, color, linewidth=0.8, alpha=0.8):
    axis.plot(
        values[:, 0], values[:, 1], values[:, 2],
        color=color, linewidth=linewidth, alpha=alpha,
    )


def export_overlay_png(path: Path, trace: EarTrace) -> None:
    views = (
        (18, -65, "view 1"),
        (18, 65, "view 2"),
        (90, -90, "superior"),
        (0, 0, "side"),
    )
    figure = plt.figure(figsize=(16, 4))
    all_points = np.concatenate(
        [
            np.asarray(trace.prepared.crop_mesh.vertices),
            trace.ground_truth_world,
            trace.final_world,
        ]
    )
    refining = not np.array_equal(trace.coarse_local, trace.final_local)
    for panel, (elevation, azimuth, title) in enumerate(views, 1):
        axis = figure.add_subplot(1, 4, panel, projection="3d")
        _plot_mesh(axis, trace.prepared.crop_mesh)
        for _, start, end in CONTOURS:
            _plot_polyline(axis, trace.ground_truth_world[start:end], "#24a058")
            _plot_polyline(axis, trace.final_world[start:end], "#dc3737")
        axis.scatter(*trace.ground_truth_world.T, s=8, c="#24a058", label="ground truth")
        if refining:
            axis.scatter(*trace.coarse_world.T, s=6, c="#f0962d", label="coarse")
        axis.scatter(*trace.final_world.T, s=8, c="#dc3737", label="raw final")
        if trace.projected_world is not None:
            axis.scatter(*trace.projected_world.T, s=6, c="#19afcd", label="projected")
        for ground_truth, prediction in zip(
            trace.ground_truth_world, trace.final_world
        ):
            axis.plot(
                *np.stack([ground_truth, prediction]).T,
                color="#777777", linewidth=0.35, alpha=0.5,
            )
        if refining:
            for coarse, final in zip(trace.coarse_world, trace.final_world):
                axis.plot(
                    *np.stack([coarse, final]).T,
                    color="#f0962d", linewidth=0.45, alpha=0.6,
                )
        if trace.projected_world is not None:
            for raw, projected in zip(trace.final_world, trace.projected_world):
                axis.plot(
                    *np.stack([raw, projected]).T,
                    color="#19afcd", linewidth=0.4, alpha=0.6,
                )
        axis.scatter(
            *trace.predicted_center_world,
            s=28, c="#874bb9", marker="x", label="predicted centre",
        )
        axis.scatter(
            *trace.true_center_world,
            s=24, c="#111111", marker="+", label="true centre",
        )
        for edge in _box_edges(trace.prepared.primary_box.for_ear(trace.ear)):
            axis.plot(*np.stack(edge).T, color="#874bb9", linewidth=0.45, alpha=0.5)
        for edge in _box_edges(trace.prepared.backup_box.for_ear(trace.ear)):
            axis.plot(*np.stack(edge).T, color="#777777", linewidth=0.3, alpha=0.3)
        _set_equal_axes(axis, all_points)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title)
        axis.set_axis_off()
        if panel == 1:
            axis.legend(loc="upper left", fontsize=7)
    figure.suptitle(
        f"{trace.subject_id} {trace.ear} | raw MD {trace.raw_errors_mm.mean():.3f} mm"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _colored_mesh(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    result.visual.face_colors = np.tile(color, (len(result.faces), 1))
    return result


def _marker_mesh(
    points: np.ndarray, color: np.ndarray, radius: float
) -> Optional[trimesh.Trimesh]:
    meshes = []
    for point in np.asarray(points):
        marker = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        marker.apply_translation(point)
        marker.visual.face_colors = np.tile(color, (len(marker.faces), 1))
        meshes.append(marker)
    return trimesh.util.concatenate(meshes) if meshes else None


def _line_mesh(segments, color: np.ndarray, radius: float) -> Optional[trimesh.Trimesh]:
    meshes = []
    for start, end in segments:
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            continue
        transform = trimesh.geometry.align_vectors(
            [0.0, 0.0, 1.0], direction / length
        )
        if transform is None:
            transform = np.eye(4)
        transform[:3, 3] = (start + end) * 0.5
        cylinder = trimesh.creation.cylinder(
            radius=radius,
            height=length,
            sections=6,
            transform=transform,
        )
        cylinder.visual.face_colors = np.tile(color, (len(cylinder.faces), 1))
        meshes.append(cylinder)
    return trimesh.util.concatenate(meshes) if meshes else None


def export_overlay_glb(path: Path, trace: EarTrace) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(
        _colored_mesh(trace.prepared.crop_mesh, GREY), geom_name="selected_crop"
    )
    extent = float(
        np.linalg.norm(np.ptp(np.asarray(trace.prepared.crop_mesh.vertices), axis=0))
    )
    marker_radius = max(extent * 0.006, 0.18)
    line_radius = max(marker_radius * 0.12, 0.025)
    layers = [
        ("ground_truth", _marker_mesh(trace.ground_truth_world, GREEN, marker_radius)),
        ("raw_final", _marker_mesh(trace.final_world, RED, marker_radius)),
    ]
    if not np.array_equal(trace.coarse_local, trace.final_local):
        layers.append(
            ("coarse", _marker_mesh(trace.coarse_world, ORANGE, marker_radius * 0.8))
        )
    if trace.projected_world is not None:
        layers.append(
            (
                "projected",
                _marker_mesh(trace.projected_world, CYAN, marker_radius * 0.75),
            )
        )
    for name, geometry in layers:
        if geometry is not None:
            scene.add_geometry(geometry, geom_name=name)
    centres = (
        ("predicted_centre", trace.predicted_center_world, PURPLE),
        ("true_centre", trace.true_center_world, np.array([30, 30, 30, 255], dtype=np.uint8)),
    )
    for name, point, color in centres:
        geometry = _marker_mesh(np.asarray(point)[None, :], color, marker_radius * 1.5)
        if geometry is not None:
            scene.add_geometry(geometry, geom_name=name)
    for contour_name, start, end in CONTOURS:
        for prefix, values, color in (
            ("ground_truth", trace.ground_truth_world, GREEN),
            ("raw_final", trace.final_world, RED),
        ):
            geometry = _line_mesh(
                zip(values[start : end - 1], values[start + 1 : end]),
                color,
                line_radius,
            )
            if geometry is not None:
                scene.add_geometry(
                    geometry, geom_name=f"{prefix}_{contour_name}_contour"
                )
    correspondence = _line_mesh(
        zip(trace.ground_truth_world, trace.final_world), GREY, line_radius
    )
    if correspondence is not None:
        scene.add_geometry(correspondence, geom_name="prediction_error_vectors")
    if not np.array_equal(trace.coarse_local, trace.final_local):
        refinement = _line_mesh(
            zip(trace.coarse_world, trace.final_world), ORANGE, line_radius
        )
        if refinement is not None:
            scene.add_geometry(refinement, geom_name="refinement_vectors")
    if trace.projected_world is not None:
        projection = _line_mesh(
            zip(trace.final_world, trace.projected_world), CYAN, line_radius
        )
        if projection is not None:
            scene.add_geometry(projection, geom_name="projection_vectors")
    for name, box, color in (
        ("primary_box", trace.prepared.primary_box.for_ear(trace.ear), PURPLE),
        ("backup_box", trace.prepared.backup_box.for_ear(trace.ear), GREY),
    ):
        geometry = _line_mesh(_box_edges(box), color, line_radius)
        if geometry is not None:
            scene.add_geometry(geometry, geom_name=name)
    scene.export(path)


def _surface_scores(
    context: FoldContext, trace: EarTrace, scores: np.ndarray
):
    if context.backbone == "meshnet":
        return trace.simplified_mesh.copy(), np.asarray(scores, dtype=np.float32), "face"
    mesh = trace.prepared.crop_mesh.copy()
    count = min(4, len(trace.input_xyz_world))
    distances, indices = cKDTree(trace.input_xyz_world).query(
        np.asarray(mesh.vertices), k=count
    )
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    vertex_scores = np.sum(
        np.asarray(scores)[indices] * weights, axis=1
    ).astype(np.float32)
    return mesh, vertex_scores, "vertex"


def export_importance_artifacts(
    base_path: Path,
    context: FoldContext,
    trace: EarTrace,
    scores: np.ndarray,
    signed: bool,
) -> None:
    colors = colorize_signed(scores) if signed else colorize_importance(scores)
    ply_path = base_path.with_suffix(".ply")
    if context.backbone == "meshnet":
        exact = trace.simplified_mesh.copy()
        exact.visual.face_colors = colors
        exact.export(ply_path)
    else:
        trimesh.points.PointCloud(
            trace.input_xyz_world, colors=colors
        ).export(ply_path)

    surface, surface_values, mode = _surface_scores(context, trace, scores)
    surface_colors = (
        colorize_signed(surface_values)
        if signed
        else colorize_importance(surface_values)
    )
    if mode == "face":
        surface.visual.face_colors = surface_colors
    else:
        surface.visual.vertex_colors = surface_colors
    trimesh.Scene([surface]).export(base_path.with_suffix(".glb"))

    views = ((18, -65), (18, 65), (90, -90), (0, 0))
    figure = plt.figure(figsize=(16, 4))
    for panel, (elevation, azimuth) in enumerate(views, 1):
        axis = figure.add_subplot(1, 4, panel, projection="3d")
        if context.backbone == "meshnet":
            _plot_mesh(axis, trace.simplified_mesh, colors, alpha=1.0)
        else:
            axis.scatter(
                trace.input_xyz_world[:, 0],
                trace.input_xyz_world[:, 1],
                trace.input_xyz_world[:, 2],
                s=1.0,
                c=colors[:, :3] / 255.0,
                depthshade=False,
            )
        _set_equal_axes(axis, trace.input_xyz_world)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
    figure.suptitle(base_path.name.replace("importance_", "").replace("_", " "))
    figure.tight_layout()
    figure.savefig(base_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_landmark_csv(path: Path, trace: EarTrace) -> None:
    projected = (
        trace.projected_world
        if trace.projected_world is not None
        else trace.final_world
    )
    projected_errors = (
        trace.projected_errors_mm
        if trace.projected_errors_mm is not None
        else trace.raw_errors_mm
    )
    fields = [
        "index", "contour", "anchor",
        *[f"ground_truth_{axis}" for axis in "xyz"],
        *[f"coarse_{axis}" for axis in "xyz"],
        *[f"raw_final_{axis}" for axis in "xyz"],
        *[f"projected_{axis}" for axis in "xyz"],
        *[f"raw_to_ground_truth_d{axis}" for axis in "xyz"],
        *[f"refinement_d{axis}" for axis in "xyz"],
        *[f"projection_d{axis}" for axis in "xyz"],
        "coarse_error_mm", "raw_error_mm", "projected_error_mm",
        "refinement_distance_mm", "projection_distance_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(85):
            row = {
                "index": index,
                "contour": CONTOUR_BY_INDEX[index],
                "anchor": index in ANCHOR_INDICES,
            }
            for prefix, values in (
                ("ground_truth", trace.ground_truth_world[index]),
                ("coarse", trace.coarse_world[index]),
                ("raw_final", trace.final_world[index]),
                ("projected", projected[index]),
            ):
                for axis, value in zip("xyz", values):
                    row[f"{prefix}_{axis}"] = float(value)
            for prefix, values in (
                (
                    "raw_to_ground_truth_d",
                    trace.final_world[index] - trace.ground_truth_world[index],
                ),
                ("refinement_d", trace.final_world[index] - trace.coarse_world[index]),
                ("projection_d", projected[index] - trace.final_world[index]),
            ):
                for axis, value in zip("xyz", values):
                    row[f"{prefix}{axis}"] = float(value)
            row.update(
                coarse_error_mm=float(trace.coarse_errors_mm[index]),
                raw_error_mm=float(trace.raw_errors_mm[index]),
                projected_error_mm=float(projected_errors[index]),
                refinement_distance_mm=float(
                    np.linalg.norm(trace.final_world[index] - trace.coarse_world[index])
                ),
                projection_distance_mm=float(
                    np.linalg.norm(projected[index] - trace.final_world[index])
                ),
            )
            writer.writerow(row)


def _contour_metrics(errors: np.ndarray) -> Mapping[str, float]:
    return {
        name: float(np.mean(errors[start:end]))
        for name, start, end in CONTOURS
    }


def render_trace(
    context: FoldContext,
    trace: EarTrace,
    output_root: Path,
    cluster_size: int,
    occlusion_batch_size: int,
) -> Mapping[str, object]:
    output = output_root / f"{trace.subject_id}_{trace.ear}"
    output.mkdir(parents=True, exist_ok=True)
    gradients = gradient_importance(context, trace)
    occlusion = occlusion_importance(
        context,
        trace,
        cluster_size=cluster_size,
        batch_size=occlusion_batch_size,
    )
    importance = {**gradients, **occlusion}
    export_overlay_png(output / "overlay.png", trace)
    export_overlay_glb(output / "overlay.glb", trace)
    for name, signed in (
        ("gradient_xyz", False),
        ("gradient_normals", False),
        ("occlusion_md", True),
        ("occlusion_displacement", False),
    ):
        export_importance_artifacts(
            output / f"importance_{name}",
            context,
            trace,
            importance[name],
            signed,
        )
    _write_landmark_csv(output / "landmarks.csv", trace)
    projected = (
        trace.projected_world
        if trace.projected_world is not None
        else trace.final_world
    )
    projected_errors = (
        trace.projected_errors_mm
        if trace.projected_errors_mm is not None
        else trace.raw_errors_mm
    )
    np.savez_compressed(
        output / "arrays.npz",
        input_features=trace.input_features,
        input_xyz_official_mm=trace.input_xyz_world,
        neighbors=(
            trace.neighbors
            if trace.neighbors is not None
            else np.empty((0, 3), dtype=np.int64)
        ),
        ground_truth_official_mm=trace.ground_truth_world,
        coarse_official_mm=trace.coarse_world,
        raw_final_official_mm=trace.final_world,
        projected_official_mm=projected,
        predicted_center_canonical_mm=trace.prepared.center,
        true_center_canonical_mm=trace.true_center_canonical,
        primary_box_min_canonical_mm=trace.prepared.primary_box.minimum,
        primary_box_max_canonical_mm=trace.prepared.primary_box.maximum,
        backup_box_min_canonical_mm=trace.prepared.backup_box.minimum,
        backup_box_max_canonical_mm=trace.prepared.backup_box.maximum,
        gradient_xyz=importance["gradient_xyz"],
        gradient_normals=importance["gradient_normals"],
        occlusion_md_delta_mm=importance["occlusion_md"],
        occlusion_prediction_displacement_mm=importance[
            "occlusion_displacement"
        ],
        occlusion_cluster_labels=importance["cluster_labels"],
    )
    manifest = {
        "schema_version": 1,
        "subject_id": trace.subject_id,
        "ear": trace.ear,
        "outer_fold": context.outer_fold,
        "held_out": True,
        "checkpoint": str(context.checkpoint_path),
        "checkpoint_sha256": file_sha256(context.checkpoint_path),
        "artifact_provenance": context.provenance_level,
        "artifact_sha256": {
            "folds_json": file_sha256(context.folds_path),
            "predictions_json": file_sha256(context.predictions_path),
            "calibration_json": file_sha256(context.calibration_path),
        },
        "run_seed": context.run_seed,
        "validation_sample_seed": trace.sample_seed,
        "backbone": context.backbone,
        "model_config": context.model_config,
        "num_points": context.num_points,
        "coordinate_frame": "official_head_mm",
        "canonical_right_reflection": [1.0, -1.0, 1.0],
        "crop": {
            "used_backup": bool(trace.prepared.crop_stats["used_backup"]),
            "selected": (
                "backup" if trace.prepared.crop_stats["used_backup"] else "primary"
            ),
            "geometry": trace.prepared.crop_stats,
            "primary_complete": bool(
                trace.prepared.primary_box.contains(
                    trace.prepared.canonical_landmarks
                ).all()
            ),
            "backup_complete": bool(
                trace.prepared.backup_box.contains(
                    trace.prepared.canonical_landmarks
                ).all()
            ),
            "local_scale_mm": float(trace.prepared.transform.scale),
        },
        "centre_error_mm": float(
            np.linalg.norm(trace.prepared.center - trace.true_center_canonical)
        ),
        "metrics": {
            "coarse_md_mm": float(np.mean(trace.coarse_errors_mm)),
            "raw_md_mm": float(np.mean(trace.raw_errors_mm)),
            "projected_md_mm": float(np.mean(projected_errors)),
            "raw_anchor_md_mm": float(
                np.mean(trace.raw_errors_mm[list(ANCHOR_INDICES)])
            ),
            "raw_contour_md_mm": _contour_metrics(trace.raw_errors_mm),
            "mean_refinement_distance_mm": float(
                np.mean(np.linalg.norm(trace.final_world - trace.coarse_world, axis=1))
            ),
            "mean_projection_distance_mm": float(
                np.mean(np.linalg.norm(projected - trace.final_world, axis=1))
            ),
        },
        "importance": {
            "objective": "raw official mean Euclidean landmark distance in millimetres",
            "cluster_target_size": int(cluster_size),
            "cluster_count": int(len(np.unique(importance["cluster_labels"]))),
            "occlusion_mask": "sample-wide mean feature vector",
            "projection_excluded_from_objective": True,
        },
        "outputs": {
            "landmarks_csv": "landmarks.csv",
            "arrays_npz": "arrays.npz",
            "overlay_png": "overlay.png",
            "overlay_glb": "overlay.glb",
            "importance": {
                name: {
                    "png": f"importance_{name}.png",
                    "glb": f"importance_{name}.glb",
                    "ply": f"importance_{name}.ply",
                }
                for name in (
                    "gradient_xyz",
                    "gradient_normals",
                    "occlusion_md",
                    "occlusion_displacement",
                )
            },
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(
        f"Wrote {trace.subject_id}:{trace.ear} "
        f"raw MD={np.mean(trace.raw_errors_mm):.6f} mm to {output}"
    )
    return manifest


def command_single(args: argparse.Namespace) -> None:
    context = load_fold_context(args)
    ears = EAR_NAMES if args.ear == "both" else (args.ear,)
    output = Path(args.output_dir)
    summaries = []
    for ear in ears:
        trace = prepare_trace(context, args.subject_id, ear, include_projection=True)
        summaries.append(
            render_trace(
                context,
                trace,
                output,
                args.occlusion_cluster_size,
                args.occlusion_batch_size,
            )
        )
    _write_json(output / f"{args.subject_id}_summary.json", {"cases": summaries})


def _gallery_selection(records):
    ordered = sorted(
        records,
        key=lambda item: (
            item["raw_md_mm"], item["subject_id"], item["ear"]
        ),
    )
    median_value = float(np.median([item["raw_md_mm"] for item in ordered]))
    middle = min(
        ordered,
        key=lambda item: (
            abs(item["raw_md_mm"] - median_value),
            item["subject_id"],
            item["ear"],
        ),
    )
    worst = min(
        records,
        key=lambda item: (
            -item["raw_md_mm"], item["subject_id"], item["ear"]
        ),
    )
    return {"best": ordered[0], "median": middle, "worst": worst}


def _write_gallery_csv(path: Path, records, selected) -> None:
    roles = {
        (item["subject_id"], item["ear"]): role
        for role, item in selected.items()
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("subject_id", "ear", "raw_md_mm", "role")
        )
        writer.writeheader()
        for item in sorted(
            records,
            key=lambda value: (
                value["raw_md_mm"], value["subject_id"], value["ear"]
            ),
        ):
            writer.writerow(
                {
                    **item,
                    "role": roles.get((item["subject_id"], item["ear"]), ""),
                }
            )


def _gallery_contact_sheet(path: Path, output: Path, selected) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, role in zip(axes, ("best", "median", "worst")):
        item = selected[role]
        image = plt.imread(
            output / f"{item['subject_id']}_{item['ear']}" / "overlay.png"
        )
        axis.imshow(image)
        axis.set_title(
            f"{role}: {item['subject_id']} {item['ear']}\n"
            f"{item['raw_md_mm']:.3f} mm"
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def command_gallery(args: argparse.Namespace) -> None:
    context = load_fold_context(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(context.validation_ids) * 2
    completed = 0
    for subject_id in context.validation_ids:
        for ear in EAR_NAMES:
            trace = prepare_trace(
                context, subject_id, ear, include_projection=False
            )
            records.append(
                {
                    "subject_id": subject_id,
                    "ear": ear,
                    "raw_md_mm": float(np.mean(trace.raw_errors_mm)),
                }
            )
            completed += 1
            print(f"Scored held-out ear {completed}/{total}: {subject_id}:{ear}")
    selected = _gallery_selection(records)
    rendered = {}
    for role in ("best", "median", "worst"):
        item = selected[role]
        trace = prepare_trace(
            context,
            item["subject_id"],
            item["ear"],
            include_projection=True,
        )
        rendered[role] = render_trace(
            context,
            trace,
            output,
            args.occlusion_cluster_size,
            args.occlusion_batch_size,
        )
    _write_gallery_csv(output / "gallery_summary.csv", records, selected)
    _gallery_contact_sheet(
        output / "gallery_contact_sheet.png", output, selected
    )
    _write_json(
        output / "gallery_manifest.json",
        {
            "outer_fold": context.outer_fold,
            "ranking_metric": "raw per-ear mean distance in millimetres",
            "case_count": len(records),
            "selected": selected,
            "rendered": rendered,
        },
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--mesh-dir", default=str(DATA_ROOT / "mesh"))
    parser.add_argument("--landmarks-dir", default=str(DATA_ROOT / "landmarks"))
    parser.add_argument(
        "--run-seed",
        type=int,
        help="Required for existing checkpoints; future checkpoints store this value.",
    )
    parser.add_argument("--occlusion-cluster-size", type=int, default=128)
    parser.add_argument("--occlusion-batch-size", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser(
        "single", help="Render one held-out subject and ear selection"
    )
    _add_common_arguments(single)
    single.add_argument("--subject-id", required=True)
    single.add_argument(
        "--ear", choices=("left", "right", "both"), required=True
    )
    single.set_defaults(function=command_single)
    gallery = commands.add_parser(
        "gallery", help="Render held-out best/median/worst ears"
    )
    _add_common_arguments(gallery)
    gallery.set_defaults(function=command_gallery)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
