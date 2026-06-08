"""Ear crop utilities for two-branch PointNet++ training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
) -> np.ndarray:
    """Sample normalized point features from one crop box."""
    crop_mesh = filter_mesh_by_normalized_box(mesh, transform, crop_box)
    point_features = sample_mesh_surface(crop_mesh, num_points=num_points, seed=seed)
    point_features = normalize_point_features(point_features, transform)
    if mirror_y:
        point_features[:, 1] *= -1.0
        if point_features.shape[1] >= 5:
            point_features[:, 4] *= -1.0
    return point_features.astype(np.float32)


def export_subject_crop_plys(
    dataset: MeshLandmarkDataset,
    subject_ids: Sequence[str],
    crop_config: Mapping[str, CropBox],
    output_dir: str,
    split_name: str,
) -> None:
    """Save all left/right crop meshes for visual inspection."""
    id_to_index = {dataset.get_identifier(idx): idx for idx in range(len(dataset))}
    split_dir = Path(output_dir) / "crops" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subject_ids:
        if subject_id not in id_to_index:
            raise ValueError(f"Unknown subject id: {subject_id}")
        mesh, _, _ = dataset[id_to_index[subject_id]]
        transform = compute_mesh_normalization(mesh)
        for ear in EAR_NAMES:
            crop_mesh = filter_mesh_by_normalized_box(mesh, transform, crop_config[ear])
            crop_mesh.export(split_dir / f"{subject_id}_{ear}.ply")


def make_crop_target(
    left_landmarks: np.ndarray,
    right_landmarks: np.ndarray,
    transform: MeshNormalization,
) -> np.ndarray:
    return make_landmark_target(left_landmarks, right_landmarks, transform)
