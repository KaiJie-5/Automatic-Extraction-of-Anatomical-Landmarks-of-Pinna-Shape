"""Repository-native closest-point projection onto triangular meshes."""

from __future__ import annotations

import numpy as np
import trimesh


def _closest_on_segments(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    direction = ends - starts
    denominator = np.einsum("ij,ij->i", direction, direction)
    t = np.divide(
        np.einsum("ij,ij->i", point - starts, direction),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-20,
    )
    return starts + np.clip(t, 0.0, 1.0)[:, None] * direction


def _project_one_with_face(
    point: np.ndarray, triangles: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, int]:
    best_point = None
    best_distance = np.inf
    best_face = -1
    for start in range(0, len(triangles), chunk_size):
        tri = triangles[start : start + chunk_size]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        candidates = [
            (_closest_on_segments(point, a, b), np.arange(len(tri))),
            (_closest_on_segments(point, b, c), np.arange(len(tri))),
            (_closest_on_segments(point, c, a), np.arange(len(tri))),
        ]
        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        normal_sq = np.einsum("ij,ij->i", normal, normal)
        offset = np.divide(
            np.einsum("ij,ij->i", point - a, normal),
            normal_sq,
            out=np.zeros_like(normal_sq),
            where=normal_sq > 1e-20,
        )
        plane = point - offset[:, None] * normal
        v0, v1, v2 = ab, ac, plane - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denominator = d00 * d11 - d01 * d01
        u = np.divide(d11 * d20 - d01 * d21, denominator, out=np.full_like(denominator, -1.0), where=np.abs(denominator) > 1e-20)
        v = np.divide(d00 * d21 - d01 * d20, denominator, out=np.full_like(denominator, -1.0), where=np.abs(denominator) > 1e-20)
        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        if np.any(inside):
            candidates.append((plane[inside], np.flatnonzero(inside)))
        for candidate, local_faces in candidates:
            distances = np.einsum("ij,ij->i", candidate - point, candidate - point)
            index = int(np.argmin(distances))
            if distances[index] < best_distance:
                best_distance = float(distances[index])
                best_point = candidate[index].copy()
                best_face = start + int(local_faces[index])
    if best_point is None:
        raise ValueError("mesh has no usable triangles")
    return best_point, best_face


def _project_one(point: np.ndarray, triangles: np.ndarray, chunk_size: int) -> np.ndarray:
    best_point = None
    best_distance = np.inf
    for start in range(0, len(triangles), chunk_size):
        tri = triangles[start : start + chunk_size]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        candidates = [
            _closest_on_segments(point, a, b),
            _closest_on_segments(point, b, c),
            _closest_on_segments(point, c, a),
        ]
        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        normal_sq = np.einsum("ij,ij->i", normal, normal)
        offset = np.divide(
            np.einsum("ij,ij->i", point - a, normal),
            normal_sq,
            out=np.zeros_like(normal_sq),
            where=normal_sq > 1e-20,
        )
        plane = point - offset[:, None] * normal
        v0, v1, v2 = ab, ac, plane - a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", v2, v0)
        d21 = np.einsum("ij,ij->i", v2, v1)
        denominator = d00 * d11 - d01 * d01
        u = np.divide(
            d11 * d20 - d01 * d21,
            denominator,
            out=np.full_like(denominator, -1.0),
            where=np.abs(denominator) > 1e-20,
        )
        v = np.divide(
            d00 * d21 - d01 * d20,
            denominator,
            out=np.full_like(denominator, -1.0),
            where=np.abs(denominator) > 1e-20,
        )
        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        if np.any(inside):
            candidates.append(plane[inside])
        for candidate in candidates:
            distances = np.einsum("ij,ij->i", candidate - point, candidate - point)
            index = int(np.argmin(distances))
            if distances[index] < best_distance:
                best_distance = float(distances[index])
                best_point = candidate[index].copy()
    if best_point is None:
        raise ValueError("mesh has no usable triangles")
    return best_point


def project_points_to_mesh(
    points: np.ndarray, mesh: trimesh.Trimesh, chunk_size: int = 65536
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("projection mesh must contain triangular faces")
    triangles = vertices[faces]
    return np.stack([_project_one(point, triangles, chunk_size) for point in values]).astype(np.float32)


def project_points_to_mesh_with_faces(
    points: np.ndarray, mesh: trimesh.Trimesh, chunk_size: int = 65536
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact closest points and deterministic source face indices."""
    values = np.asarray(points, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("projection mesh must contain triangular faces")
    triangles = vertices[faces]
    projected = [
        _project_one_with_face(point, triangles, chunk_size) for point in values
    ]
    return (
        np.stack([item[0] for item in projected]).astype(np.float32),
        np.asarray([item[1] for item in projected], dtype=np.int64),
    )
