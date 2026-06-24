"""Ear crop utilities for single-ear PointNet++ training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import torch
import numpy as np
import trimesh
from trimesh import Trimesh

from .dataset import Dataset as MeshLandmarkDataset
from .preprocessing import (
    MeshNormalization,
    compute_mesh_normalization,
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


def _slice_mesh_plane(
    mesh: Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> Trimesh:
    if hasattr(mesh, "slice_plane"):
        return mesh.slice_plane(
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            cap=False,
        )
    return trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        cap=False,
    )


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
    """Return a mesh clipped to the normalized crop box."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        normalized_vertices = transform.normalize_xyz(vertices)
        mask = points_inside_box(normalized_vertices, crop_box)
        selected = vertices[mask]
        return Trimesh(vertices=selected, faces=np.empty((0, 3), dtype=np.int64), process=False)

    minimum = transform.denormalize_xyz(crop_box.minimum.astype(np.float32))
    maximum = transform.denormalize_xyz(crop_box.maximum.astype(np.float32))
    crop_mesh = mesh.copy()
    axes = np.eye(3, dtype=np.float64)
    bounds = ((minimum, axes), (maximum, -axes))

    for origins, normals in bounds:
        for axis_idx in range(3):
            if len(crop_mesh.vertices) == 0:
                break
            sliced = _slice_mesh_plane(
                crop_mesh,
                plane_origin=origins[axis_idx] * axes[axis_idx],
                plane_normal=normals[axis_idx],
            )
            if not isinstance(sliced, Trimesh):
                return Trimesh(
                    vertices=np.empty((0, 3), dtype=np.float32),
                    faces=np.empty((0, 3), dtype=np.int64),
                    process=False,
                )
            crop_mesh = sliced

    if not isinstance(crop_mesh, Trimesh):
        return Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64))
    crop_mesh.remove_unreferenced_vertices()
    return crop_mesh


def _validate_crop_mesh(crop_mesh: Trimesh) -> None:
    vertices = np.asarray(crop_mesh.vertices, dtype=np.float32)
    faces = np.asarray(crop_mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(
            "The clipped ear crop mesh is empty. Increase --crop-margin or check crop_config."
        )
    if faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0:
        return


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
    """Clip an ear submesh first, then sample normalized point features from it."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    crop_mesh = filter_mesh_by_normalized_box(mesh, transform, crop_box)
    _validate_crop_mesh(crop_mesh)
    point_features = sample_mesh_surface(crop_mesh, num_points=num_points, seed=seed)
    point_features = normalize_point_features(point_features, transform)
    if mirror_y:
        point_features[:, 1] *= -1.0
        if point_features.shape[1] >= 5:
            point_features[:, 4] *= -1.0

    stats = {
        "requested_points": int(num_points),
        "candidate_count": int(num_points),
        "inside_count": int(num_points),
        "inside_ratio": 1.0,
        "attempts": 1,
        "final_sampled_count": int(point_features.shape[0]),
        "used_replacement": bool(len(crop_mesh.faces) == 0 and len(crop_mesh.vertices) < num_points),
        "mirrored": bool(mirror_y),
        "low_inside_ratio": False,
        "crop_vertex_count": int(len(crop_mesh.vertices)),
        "crop_face_count": int(len(crop_mesh.faces)),
        "crop_first_sampling": True,
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
    landmarks: np.ndarray,
    transform: MeshNormalization,
) -> np.ndarray:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (85, 3):
        raise ValueError("ear landmarks must have shape (85, 3)")
    return transform.normalize_xyz(landmarks).astype(np.float32)

def make_box_target(
    landmarks: np.ndarray,
    transform: MeshNormalization,
    margin: float = 0.15,
) -> np.ndarray:
    """Creates a [cx, cy, cz, sx, sy, sz] target array from GT landmarks."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (85, 3):
        raise ValueError("ear landmarks must have shape (85, 3)")

    normalized = transform.normalize_xyz(landmarks)
    minimum = normalized.min(axis=0)
    maximum = normalized.max(axis=0)

    # Assuming expand_box returns a CropBox object with .minimum and .maximum
    box = expand_box(minimum, maximum, margin)
    center = (box.minimum + box.maximum) * 0.5
    size = box.maximum - box.minimum

    return np.concatenate([center, size]).astype(np.float32)

def crop_box_from_center_extents(center, negative_extent, positive_extent, scale=1.0):
    """Creates a bounding box using asymmetric negative/positive extents."""
    center = np.asarray(center, dtype=np.float32)
    negative_extent = np.asarray(negative_extent, dtype=np.float32) * float(scale)
    positive_extent = np.asarray(positive_extent, dtype=np.float32) * float(scale)

    return CropBox(
        minimum=(center - negative_extent).astype(np.float32),
        maximum=(center + positive_extent).astype(np.float32),
    )

def compute_calibrated_asymmetric_extents(
    dataset,
    subject_ids,
    broad_crop_config,
    box_model,
    ear_points=8192,
    device="cuda",
    seed=0,
    margin=1.05,     
    percentile=98.0,  
    tta_runs=5,
    axis_multiplier=None, 
):
    """Calculates independent safety margins for the X, Y, and Z axes."""
    extents = {
        "left": {"neg": [], "pos": []},
        "right": {"neg": [], "pos": []}
    }
    id_to_index = {dataset.get_identifier(i): i for i in range(len(dataset))}

    box_model.eval()

    for subject_id in subject_ids:
        if subject_id not in id_to_index:
            continue
            
        base_idx = id_to_index[subject_id]
        mesh, left_lm, right_lm = dataset[base_idx]
        transform = compute_mesh_normalization(mesh)

        for ear, landmarks, ear_seed in [
            ("left", left_lm, seed + base_idx * 2),
            ("right", right_lm, seed + base_idx * 2 + 1),
        ]:
            centers = []

            for k in range(tta_runs):
                broad_points = sample_crop_point_features(
                    mesh=mesh,
                    transform=transform,
                    crop_box=broad_crop_config[ear],
                    num_points=ear_points,
                    seed=ear_seed + 1000 * k,
                    mirror_y=False,
                )

                points_tensor = torch.from_numpy(broad_points).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    pred_box = box_model(points_tensor).squeeze(0).cpu().numpy()

                centers.append(pred_box[:3])

            pred_center = np.median(np.stack(centers), axis=0)
            landmarks_norm = transform.normalize_xyz(landmarks.astype(np.float32))

            # ASYMMETRIC CALCULATION: How far back/forward do we need to reach?
            req_neg = pred_center - landmarks_norm.min(axis=0)
            req_pos = landmarks_norm.max(axis=0) - pred_center

            extents[ear]["neg"].append(np.maximum(req_neg, 0.0))
            extents[ear]["pos"].append(np.maximum(req_pos, 0.0))

    # Calculate base extents
    left_neg = (np.percentile(np.stack(extents["left"]["neg"]), percentile, axis=0) * margin).astype(np.float32)
    left_pos = (np.percentile(np.stack(extents["left"]["pos"]), percentile, axis=0) * margin).astype(np.float32)
    right_neg = (np.percentile(np.stack(extents["right"]["neg"]), percentile, axis=0) * margin).astype(np.float32)
    right_pos = (np.percentile(np.stack(extents["right"]["pos"]), percentile, axis=0) * margin).astype(np.float32)

    # --- Apply the axis multiplier if provided ---
    if axis_multiplier is not None:
        multiplier_arr = np.array(axis_multiplier, dtype=np.float32)
        left_neg *= multiplier_arr
        left_pos *= multiplier_arr
        right_neg *= multiplier_arr
        right_pos *= multiplier_arr

    return {
        "left": {"neg": left_neg, "pos": left_pos},
        "right": {"neg": right_neg, "pos": right_pos}
    }