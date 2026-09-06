"""Mesh sampling and coordinate normalization for landmark regression."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from trimesh import Trimesh


EPSILON = 1e-8


@dataclass(frozen=True)
class MeshNormalization:
    """Per-mesh normalization transform."""

    centroid: np.ndarray
    scale: float

    def normalize_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self.centroid) / self.scale

    def denormalize_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return xyz * self.scale + self.centroid


@dataclass(frozen=True)
class SurfaceSamples:
    """Deterministic surface samples plus their source mesh faces.

    ``sample_mesh_surface`` intentionally keeps its historical array-only API.
    Geodesic supervision uses this detailed form to map every resampled point
    back to the exact triangle on which it was generated.
    """

    features: np.ndarray
    face_indices: np.ndarray
    barycentric: np.ndarray


def compute_mesh_normalization(mesh: Trimesh) -> MeshNormalization:
    """Normalize around the full mesh centroid using max vertex radius."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh must contain vertices with shape (N, 3)")

    centroid = vertices.mean(axis=0)
    distances = np.linalg.norm(vertices - centroid, axis=1)
    scale = float(np.max(distances))
    if scale < EPSILON:
        scale = 1.0
    return MeshNormalization(centroid=centroid.astype(np.float32), scale=scale)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    return vectors / norms


def _sample_vertices(mesh: Trimesh, num_points: int, rng: np.random.Generator) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError("Cannot sample an empty mesh")
    replace = len(vertices) < num_points
    indices = rng.choice(len(vertices), size=num_points, replace=replace)
    xyz = vertices[indices]
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != vertices.shape:
        normals = np.zeros_like(xyz)
        normals[:, 2] = 1.0
    else:
        normals = normals[indices]
    return np.concatenate([xyz, _normalize_vectors(normals)], axis=1).astype(np.float32)


def sample_mesh_surface(
    mesh: Trimesh, num_points: int = 16384, seed: Optional[int] = None
) -> np.ndarray:
    """Uniformly sample points on mesh faces and attach face normals.

    Returns:
        Array with shape (num_points, 6): xyz followed by normal xyz.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    try:
        return sample_mesh_surface_with_metadata(mesh, num_points, seed).features
    except ValueError as error:
        if "surface sampling metadata requires" not in str(error):
            raise
        return _sample_vertices(mesh, num_points, np.random.default_rng(seed))


def sample_mesh_surface_with_metadata(
    mesh: Trimesh, num_points: int = 16384, seed: Optional[int] = None
) -> SurfaceSamples:
    """Sample faces exactly as :func:`sample_mesh_surface` and retain mapping.

    Face indices are indices into ``mesh.faces`` and barycentric rows correspond
    to those faces.  A valid triangular surface is required because vertex-only
    sampling has no unambiguous surface-geodesic mapping.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    rng = np.random.default_rng(seed)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("surface sampling metadata requires triangular faces")

    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = areas > EPSILON
    if not np.any(valid):
        raise ValueError("surface sampling metadata requires nondegenerate faces")

    probabilities = areas / areas.sum()
    face_indices = rng.choice(len(faces), size=num_points, replace=True, p=probabilities)
    chosen = triangles[face_indices]

    u = rng.random(num_points, dtype=np.float32)
    v = rng.random(num_points, dtype=np.float32)
    sqrt_u = np.sqrt(u)
    bary0 = 1.0 - sqrt_u
    bary1 = sqrt_u * (1.0 - v)
    bary2 = sqrt_u * v
    xyz = (
        chosen[:, 0] * bary0[:, None]
        + chosen[:, 1] * bary1[:, None]
        + chosen[:, 2] * bary2[:, None]
    )

    normals = _normalize_vectors(cross[face_indices].astype(np.float32))
    features = np.concatenate([xyz, normals], axis=1).astype(np.float32)
    barycentric = np.stack([bary0, bary1, bary2], axis=1).astype(np.float32)
    return SurfaceSamples(
        features=features,
        face_indices=face_indices.astype(np.int64),
        barycentric=barycentric,
    )


def normalize_point_features(
    point_features: np.ndarray, transform: MeshNormalization
) -> np.ndarray:
    """Normalize xyz channels while keeping normal channels unit length."""
    features = np.asarray(point_features, dtype=np.float32).copy()
    if features.ndim != 2 or features.shape[1] < 3:
        raise ValueError("point_features must have shape (N, C>=3)")
    features[:, :3] = transform.normalize_xyz(features[:, :3])
    if features.shape[1] >= 6:
        features[:, 3:6] = _normalize_vectors(features[:, 3:6])
    return features


def make_landmark_target(
    left_landmarks: np.ndarray, right_landmarks: np.ndarray, transform: MeshNormalization
) -> np.ndarray:
    """Stack left then right landmarks and normalize xyz coordinates."""
    left = np.asarray(left_landmarks, dtype=np.float32)
    right = np.asarray(right_landmarks, dtype=np.float32)
    if left.shape != (85, 3) or right.shape != (85, 3):
        raise ValueError("left_landmarks and right_landmarks must both have shape (85, 3)")
    target = np.concatenate([left, right], axis=0)
    return transform.normalize_xyz(target).astype(np.float32)


def split_landmark_prediction(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 170 x 3 prediction into left and right ears."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (170, 3):
        raise ValueError("landmarks must have shape (170, 3)")
    return landmarks[:85], landmarks[85:]
