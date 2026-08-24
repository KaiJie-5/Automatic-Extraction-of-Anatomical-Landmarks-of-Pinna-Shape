"""World-coordinate mesh clipping and deterministic sampling."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import trimesh

from .canonical import WorldCropBox, canonicalize_point_features
from .preprocessing import sample_mesh_surface


def _empty_mesh() -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.empty((0, 3), dtype=np.float32),
        faces=np.empty((0, 3), dtype=np.int64),
        process=False,
    )


def clip_mesh_to_box(mesh: trimesh.Trimesh, box: WorldCropBox) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    cropped = mesh.copy()
    for axis in range(3):
        origin = np.zeros(3, dtype=np.float64)
        origin[axis] = float(box.minimum[axis])
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
        cropped = cropped.slice_plane(origin, normal, cap=False)
        if not isinstance(cropped, trimesh.Trimesh) or len(cropped.vertices) == 0:
            return _empty_mesh()
        origin[axis] = float(box.maximum[axis])
        normal[axis] = -1.0
        cropped = cropped.slice_plane(origin, normal, cap=False)
        if not isinstance(cropped, trimesh.Trimesh) or len(cropped.vertices) == 0:
            return _empty_mesh()
    cropped.remove_unreferenced_vertices()
    return cropped


def crop_geometry_stats(mesh: trimesh.Trimesh) -> dict:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    valid = (
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and len(vertices) > 0
        and np.isfinite(vertices).all()
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
        and len(faces) > 0
    )
    area = float(mesh.area) if valid and np.isfinite(mesh.area) else 0.0
    return {
        "valid": bool(valid and area > 0.0),
        "vertex_count": int(len(vertices)) if vertices.ndim else 0,
        "face_count": int(len(faces)) if faces.ndim else 0,
        "surface_area": area,
    }


def should_use_backup(stats: dict, thresholds: Optional[dict]) -> bool:
    if not stats.get("valid", False):
        return True
    if not thresholds:
        return False
    return (
        stats["face_count"] < int(thresholds.get("face_count_p01", 0))
        or stats["surface_area"] < float(thresholds.get("surface_area_p01", 0.0))
    )


def sample_canonical_crop(
    mesh: trimesh.Trimesh,
    canonical_box: WorldCropBox,
    ear: str,
    num_points: int,
    seed: int,
    fallback_box: Optional[WorldCropBox] = None,
    thresholds: Optional[dict] = None,
) -> Tuple[np.ndarray, trimesh.Trimesh, dict]:
    world_box = canonical_box.for_ear(ear)
    crop = clip_mesh_to_box(mesh, world_box)
    stats = crop_geometry_stats(crop)
    used_backup = False
    if should_use_backup(stats, thresholds):
        if fallback_box is None:
            raise ValueError("primary crop failed geometry validation and no backup crop was supplied")
        crop = clip_mesh_to_box(mesh, fallback_box.for_ear(ear))
        stats = crop_geometry_stats(crop)
        used_backup = True
    if not stats["valid"]:
        raise ValueError("ear crop is empty or has no valid triangular surface")
    features = sample_mesh_surface(crop, num_points=num_points, seed=seed)
    features = canonicalize_point_features(features, ear)
    stats = dict(stats)
    stats["used_backup"] = used_backup
    return features.astype(np.float32), crop, stats
