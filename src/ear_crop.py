"""Ear crop utilities for two-branch PointNet++ training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import trimesh
from trimesh import Trimesh

from .dataset import Dataset as MeshLandmarkDataset
from .preprocessing import (
    MeshNormalization,
    compute_mesh_normalization,
    make_landmark_target,
    normalize_point_features,
    sample_mesh_surface,
)


EAR_NAMES = ("left", "right")
EPSILON = 1e-8


@dataclass(frozen=True)
class CropBox:
    """Axis-aligned crop box in full-mesh normalized coordinates."""

    minimum: np.ndarray
    maximum: np.ndarray

    def to_dict(self) -> dict:
        return {"min": self.minimum.tolist(), "max": self.maximum.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[float]]) -> "CropBox":
        return cls(
            minimum=np.asarray(data["min"], dtype=np.float32),
            maximum=np.asarray(data["max"], dtype=np.float32),
        )


def crop_config_to_dict(crop_config: Mapping[str, CropBox]) -> dict:
    return {ear: crop_config[ear].to_dict() for ear in EAR_NAMES}


def crop_config_from_dict(data: Mapping[str, Mapping[str, Sequence[float]]]) -> Dict[str, CropBox]:
    return {ear: CropBox.from_dict(data[ear]) for ear in EAR_NAMES}


def expand_box(minimum: np.ndarray, maximum: np.ndarray, margin: float) -> CropBox:
    center = (minimum + maximum) * 0.5
    half_extent = (maximum - minimum) * 0.5
    half_extent = np.maximum(half_extent * (1.0 + float(margin)), EPSILON)
    return CropBox(
        minimum=(center - half_extent).astype(np.float32),
        maximum=(center + half_extent).astype(np.float32),
    )


def fit_crop_config_from_training_landmarks(
    dataset: MeshLandmarkDataset,
    train_subject_ids: Sequence[str],
    margin: float = 0.4,
) -> Dict[str, CropBox]:
    """Fit left/right crop boxes using training landmarks only."""
    id_to_index = {dataset.get_identifier(idx): idx for idx in range(len(dataset))}
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []

    for subject_id in train_subject_ids:
        if subject_id not in id_to_index:
            raise ValueError(f"Unknown training subject id: {subject_id}")
        mesh, left, right = dataset[id_to_index[subject_id]]
        transform = compute_mesh_normalization(mesh)
        left_points.append(transform.normalize_xyz(left.astype(np.float32)))
        right_points.append(transform.normalize_xyz(right.astype(np.float32)))

    if not left_points or not right_points:
        raise ValueError("At least one training subject is required to fit crop boxes")

    left_all = np.concatenate(left_points, axis=0)
    right_all = np.concatenate(right_points, axis=0)
    return {
        "left": expand_box(left_all.min(axis=0), left_all.max(axis=0), margin),
        "right": expand_box(right_all.min(axis=0), right_all.max(axis=0), margin),
    }


def points_inside_box(points: np.ndarray, crop_box: CropBox) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return np.all((points >= crop_box.minimum) & (points <= crop_box.maximum), axis=1)


def compute_crop_coverage(
    dataset: MeshLandmarkDataset,
    subject_ids: Sequence[str],
    crop_config: Mapping[str, CropBox],
) -> dict:
    """Report landmark coverage for crop boxes on a subject split."""
    id_to_index = {dataset.get_identifier(idx): idx for idx in range(len(dataset))}
    subject_reports = {}
    total_inside = {ear: 0 for ear in EAR_NAMES}
    total_count = {ear: 0 for ear in EAR_NAMES}

    for subject_id in subject_ids:
        if subject_id not in id_to_index:
            raise ValueError(f"Unknown subject id: {subject_id}")
        mesh, left, right = dataset[id_to_index[subject_id]]
        transform = compute_mesh_normalization(mesh)
        landmarks = {
            "left": transform.normalize_xyz(left.astype(np.float32)),
            "right": transform.normalize_xyz(right.astype(np.float32)),
        }
        subject_reports[subject_id] = {}
        for ear in EAR_NAMES:
            inside = points_inside_box(landmarks[ear], crop_config[ear])
            inside_count = int(inside.sum())
            count = int(len(inside))
            total_inside[ear] += inside_count
            total_count[ear] += count
            subject_reports[subject_id][ear] = {
                "inside": inside_count,
                "total": count,
                "coverage": inside_count / max(count, 1),
            }

    summary = {}
    for ear in EAR_NAMES:
        summary[ear] = {
            "inside": total_inside[ear],
            "total": total_count[ear],
            "coverage": total_inside[ear] / max(total_count[ear], 1),
        }

    return {"summary": summary, "subjects": subject_reports}


def filter_mesh_by_normalized_box(
    mesh: Trimesh,
    transform: MeshNormalization,
    crop_box: CropBox,
) -> Trimesh:
    """Return a submesh whose faces have at least one vertex inside the crop box."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        normalized_vertices = transform.normalize_xyz(vertices)
        mask = points_inside_box(normalized_vertices, crop_box)
        selected = vertices[mask] if np.any(mask) else vertices
        return Trimesh(vertices=selected, faces=np.empty((0, 3), dtype=np.int64), process=False)

    normalized_vertices = transform.normalize_xyz(vertices)
    vertex_mask = points_inside_box(normalized_vertices, crop_box)
    face_mask = vertex_mask[faces].any(axis=1)
    if not np.any(face_mask):
        selected = vertices[vertex_mask] if np.any(vertex_mask) else vertices
        return Trimesh(vertices=selected, faces=np.empty((0, 3), dtype=np.int64), process=False)
    parts = mesh.submesh([face_mask], append=True, repair=False)
    return parts if isinstance(parts, Trimesh) else mesh.copy()


def sample_crop_point_features(
    mesh: Trimesh,
    transform: MeshNormalization,
    crop_box: CropBox,
    num_points: int,
    seed: int,
    mirror_y: bool = False,
    oversample_factor: int = 8,
    max_attempts: int = 5,
    min_inside_ratio: float = 0.0,
    return_stats: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Sample normalized point features whose XYZ channels are inside one crop box.

    Sampling happens on the full mesh first, then the sampled points are filtered
    in normalized coordinates. This makes the returned tensor match the crop box
    more tightly than sampling from a loose face-based submesh.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if oversample_factor <= 0:
        raise ValueError("oversample_factor must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    rng = np.random.default_rng(seed)
    inside_features = None
    best_inside_features = None
    best_inside_count = 0
    inside_count = 0
    inside_ratio = 0.0
    best_inside_ratio = 0.0
    candidate_count = 0
    best_candidate_count = 0
    attempts_used = 0

    for attempt in range(max_attempts):
        attempts_used = attempt + 1
        candidate_count = int(num_points * oversample_factor * (2**attempt))
        candidate_count = max(candidate_count, num_points)
        candidates = sample_mesh_surface(mesh, num_points=candidate_count, seed=seed + attempt)
        candidates = normalize_point_features(candidates, transform)
        inside_mask = points_inside_box(candidates[:, :3], crop_box)
        inside_features = candidates[inside_mask]
        inside_count = int(inside_features.shape[0])
        inside_ratio = inside_count / max(candidate_count, 1)
        if inside_count > best_inside_count:
            best_inside_features = inside_features
            best_inside_count = inside_count
            best_inside_ratio = inside_ratio
            best_candidate_count = candidate_count
        if inside_count >= num_points:
            break

    if best_inside_features is None or best_inside_count == 0:
        raise ValueError(
            "No sampled mesh points fell inside the crop box. "
            "Increase --crop-margin or --crop-oversample-factor."
        )

    replace = best_inside_count < num_points
    selected_indices = rng.choice(best_inside_count, size=num_points, replace=replace)
    point_features = best_inside_features[selected_indices].astype(np.float32)
    if mirror_y:
        point_features[:, 1] *= -1.0
        if point_features.shape[1] >= 5:
            point_features[:, 4] *= -1.0

    stats = {
        "requested_points": int(num_points),
        "candidate_count": int(best_candidate_count),
        "inside_count": int(best_inside_count),
        "inside_ratio": float(best_inside_ratio),
        "attempts": int(attempts_used),
        "final_sampled_count": int(point_features.shape[0]),
        "used_replacement": bool(replace),
        "mirrored": bool(mirror_y),
        "low_inside_ratio": bool(best_inside_ratio < float(min_inside_ratio)),
    }
    if return_stats:
        return point_features.astype(np.float32), stats
    return point_features.astype(np.float32)


def export_point_features_ply(
    point_features: np.ndarray,
    transform: MeshNormalization,
    path: Path,
) -> None:
    """Export sampled normalized point features as an original-coordinate point cloud."""
    points = np.asarray(point_features, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("point_features must have shape (N, C>=3)")
    vertices = transform.denormalize_xyz(points[:, :3])
    point_cloud = trimesh.points.PointCloud(vertices=vertices)
    point_cloud.export(path)


def export_subject_crop_plys(
    dataset: MeshLandmarkDataset,
    subject_ids: Sequence[str],
    crop_config: Mapping[str, CropBox],
    output_dir: str,
    split_name: str,
    ear_points: int = 8192,
    seed: int = 0,
    oversample_factor: int = 8,
    max_attempts: int = 5,
    min_inside_ratio: float = 0.0,
    save_mesh: bool = True,
    save_points: bool = True,
) -> dict:
    """Save crop meshes plus exact sampled point clouds for visual inspection."""
    id_to_index = {dataset.get_identifier(idx): idx for idx in range(len(dataset))}
    split_dir = Path(output_dir) / "crops" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {}

    for subject_id in subject_ids:
        if subject_id not in id_to_index:
            raise ValueError(f"Unknown subject id: {subject_id}")
        mesh, _, _ = dataset[id_to_index[subject_id]]
        transform = compute_mesh_normalization(mesh)
        diagnostics[subject_id] = {}
        for ear in EAR_NAMES:
            if save_mesh:
                crop_mesh = filter_mesh_by_normalized_box(mesh, transform, crop_config[ear])
                crop_mesh.export(split_dir / f"{subject_id}_{ear}_mesh.ply")
            if save_points:
                point_features, stats = sample_crop_point_features(
                    mesh=mesh,
                    transform=transform,
                    crop_box=crop_config[ear],
                    num_points=ear_points,
                    seed=seed + id_to_index[subject_id] * 2 + (0 if ear == "left" else 1),
                    mirror_y=False,
                    oversample_factor=oversample_factor,
                    max_attempts=max_attempts,
                    min_inside_ratio=min_inside_ratio,
                    return_stats=True,
                )
                export_point_features_ply(
                    point_features,
                    transform,
                    split_dir / f"{subject_id}_{ear}_points.ply",
                )
                diagnostics[subject_id][ear] = stats

    return diagnostics


def make_crop_target(
    left_landmarks: np.ndarray,
    right_landmarks: np.ndarray,
    transform: MeshNormalization,
) -> np.ndarray:
    return make_landmark_target(left_landmarks, right_landmarks, transform)
