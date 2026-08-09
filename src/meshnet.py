"""MeshNet feasibility gate and portable face-feature screening model."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import trimesh

from .canonical import LocalEarTransform, canonicalize_xyz


def validate_meshnet_mesh(mesh: trimesh.Trimesh) -> tuple[bool, str]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        return False, "empty or invalid vertices"
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        return False, "empty or non-triangular faces"
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        return False, "non-finite geometry or invalid face indices"
    triangles = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(area2 <= 1e-10):
        return False, "degenerate faces"
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        return False, "non-manifold edge with more than two incident faces"
    if np.asarray(mesh.face_adjacency).ndim != 2:
        return False, "invalid face adjacency"
    return True, "ok"


def simplify_for_meshnet(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) < target_faces:
        raise ValueError(f"mesh has fewer than the requested {target_faces} faces")
    method = getattr(mesh, "simplify_quadric_decimation", None)
    if method is None:
        raise RuntimeError("trimesh simplification support is unavailable")
    try:
        simplified = method(face_count=int(target_faces))
    except TypeError:
        simplified = method(int(target_faces))
    if not isinstance(simplified, trimesh.Trimesh):
        raise ValueError("simplifier did not return a triangular mesh")
    if len(simplified.faces) != int(target_faces):
        raise ValueError(
            f"simplifier returned {len(simplified.faces)} faces instead of {int(target_faces)}"
        )
    return simplified


def run_meshnet_gate(
    named_meshes: Iterable[tuple[str, trimesh.Trimesh]],
    targets: Sequence[int] = (8192, 4096, 2048),
) -> Mapping[str, object]:
    failures = {str(target): {} for target in targets}
    common_targets = set(int(target) for target in targets)
    mesh_count = 0
    for name, mesh in named_meshes:
        mesh_count += 1
        passed_targets = set()
        for target in targets:
            try:
                candidate = simplify_for_meshnet(mesh, int(target))
                valid, reason = validate_meshnet_mesh(candidate)
                if valid:
                    passed_targets.add(int(target))
                else:
                    failures[str(target)][name] = reason
            except Exception as exc:
                failures[str(target)][name] = str(exc)
        common_targets &= passed_targets
    if mesh_count == 0:
        return {"passed": False, "target_faces": None, "failures": {"dataset": "no crops"}}
    selected = next((int(target) for target in targets if int(target) in common_targets), None)
    return {
        "passed": selected is not None,
        "target_faces": selected,
        "mesh_count": mesh_count,
        "failures": failures,
    }


def meshnet_inputs(
    mesh: trimesh.Trimesh,
    target_faces: int,
    ear: str,
    transform: LocalEarTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Simplify a crop and build MeshNet's 15D face geometry plus 3-neighbour indices."""
    simplified = simplify_for_meshnet(mesh, target_faces)
    valid, reason = validate_meshnet_mesh(simplified)
    if not valid:
        raise ValueError(f"simplified MeshNet crop is invalid: {reason}")
    vertices = transform.normalize_xyz(canonicalize_xyz(np.asarray(simplified.vertices), ear))
    faces = np.asarray(simplified.faces, dtype=np.int64)
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    corners = (triangles - centers[:, None, :]).reshape(len(faces), 9)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    normals = normals / lengths
    features = np.concatenate([centers, corners, normals], axis=1).astype(np.float32)

    neighbors = np.repeat(np.arange(len(faces), dtype=np.int64)[:, None], 3, axis=1)
    fill = np.zeros(len(faces), dtype=np.int64)
    for pair in np.asarray(simplified.face_adjacency, dtype=np.int64):
        for source, destination in ((pair[0], pair[1]), (pair[1], pair[0])):
            slot = min(int(fill[source]), 2)
            neighbors[source, slot] = destination
            fill[source] += 1
    return features, neighbors


class MeshNetLandmarkRegressor(nn.Module):
    """Small face-feature model used only after the all-crop feasibility gate passes."""

    def __init__(self, input_dim: int = 15, width: int = 128, four_heads: bool = True):
        super().__init__()
        self.four_heads = bool(four_heads)
        self.face_stem = nn.Sequential(
            nn.Conv1d(input_dim, width, 1),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.face_fusion = nn.Sequential(
            nn.Conv1d(width * 2, width * 2, 1),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
        )
        lengths = (25, 30, 20, 10) if four_heads else (85,)
        self.heads = nn.ModuleList([nn.Linear(width * 2, length * 3) for length in lengths])
        self.lengths = lengths

    def forward(self, face_features: torch.Tensor, neighbors: torch.Tensor | None = None):
        if face_features.shape[1] != 15:
            face_features = face_features.transpose(1, 2)
        features = self.face_stem(face_features)
        if neighbors is None:
            neighbor_features = features
        else:
            from .pointnet2_utils import index_points

            neighbor_features = index_points(features.transpose(1, 2), neighbors.long()).mean(dim=2)
            neighbor_features = neighbor_features.transpose(1, 2)
        pooled = self.face_fusion(torch.cat([features, neighbor_features], dim=1)).amax(dim=-1)
        outputs = [head(pooled).view(face_features.shape[0], length, 3) for head, length in zip(self.heads, self.lengths)]
        return torch.cat(outputs, dim=1)
