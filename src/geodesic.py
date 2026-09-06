"""Leakage-safe mesh-geodesic targets for surface landmark heatmaps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .pipeline_dataset import EAR_NAMES, prediction_key, prepare_ear_geometry
from .surface import project_points_to_mesh_with_faces


GEODESIC_CACHE_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_sha256(mesh) -> str:
    """Hash exact cropped vertices/faces, including shape and dtype-neutral values."""
    vertices = np.ascontiguousarray(np.asarray(mesh.vertices, dtype="<f8"))
    faces = np.ascontiguousarray(np.asarray(mesh.faces, dtype="<i8"))
    digest = hashlib.sha256()
    for values in (vertices, faces):
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def landmark_sha256(landmarks: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(landmarks, dtype="<f4"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def cache_path(root: str | Path, subject_id: str, ear: str) -> Path:
    if ear not in EAR_NAMES:
        raise ValueError(f"unsupported ear: {ear}")
    return Path(root) / f"{subject_id}_{ear}.npz"


def _mesh_edge_graph(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    _, coordinate_groups = np.unique(vertices, axis=0, return_inverse=True)
    order = np.argsort(coordinate_groups, kind="stable")
    sorted_groups = coordinate_groups[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    duplicate_edges = []
    for group in np.split(order, boundaries):
        if len(group) > 1:
            duplicate_edges.append(
                np.column_stack(
                    [np.full(len(group) - 1, group[0], dtype=np.int64), group[1:]]
                )
            )
    if duplicate_edges:
        edges = np.concatenate([edges, *duplicate_edges], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    valid = np.isfinite(lengths) & (edges[:, 0] != edges[:, 1])
    edges = edges[valid]
    # Slicing can retain distinct indices at an identical boundary position.
    # A tiny positive edge retains the intended surface connectivity because
    # scipy sparse graphs do not treat stored zero weights as traversable.
    lengths = np.maximum(lengths[valid], 1e-12)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.concatenate([lengths, lengths])
    return coo_matrix(
        (weights, (rows, columns)), shape=(len(vertices), len(vertices))
    ).tocsr()


def compute_vertex_geodesics(mesh, landmarks: np.ndarray, ear: str) -> dict:
    """Approximate continuous surface distance with an exact triangle source.

    Each annotation is first projected to its closest triangle.  Its three
    triangle vertices become virtual-source seeds with the within-triangle
    Euclidean distances as initial costs; shortest paths then follow only mesh
    edges.  Sample points on the source triangle are corrected to their direct
    within-triangle distance when the cache is consumed.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (85, 3) or not np.isfinite(points).all():
        raise ValueError("geodesic landmarks must be finite with shape (85, 3)")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices):
        raise ValueError("geodesic crop mesh has no vertices")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError("geodesic crop mesh has no triangular faces")

    projected, source_faces = project_points_to_mesh_with_faces(points, mesh)
    source_vertices = faces[source_faces]
    unique_sources, inverse = np.unique(source_vertices.reshape(-1), return_inverse=True)
    shortest = dijkstra(
        _mesh_edge_graph(mesh),
        directed=False,
        indices=unique_sources,
        return_predecessors=False,
    )
    shortest = np.atleast_2d(np.asarray(shortest, dtype=np.float64))
    source_rows = inverse.reshape(85, 3)
    initial = np.linalg.norm(
        vertices[source_vertices] - projected.astype(np.float64)[:, None, :],
        axis=-1,
    )
    fields = np.empty((85, len(vertices)), dtype=np.float32)
    for landmark_index in range(85):
        candidates = (
            shortest[source_rows[landmark_index]]
            + initial[landmark_index, :, None]
        )
        fields[landmark_index] = np.min(candidates, axis=0).astype(np.float32)

    normals = np.asarray(mesh.face_normals, dtype=np.float32)[source_faces].copy()
    if ear == "right":
        normals[:, 1] *= -1.0
    return {
        "vertex_distances_mm": fields,
        "source_projected": projected.astype(np.float32),
        "source_projection_error_mm": np.linalg.norm(
            projected.astype(np.float64) - points, axis=-1
        ).astype(np.float32),
        "source_face_indices": source_faces.astype(np.int64),
        "source_normals_canonical": normals.astype(np.float32),
    }


def save_geodesic_cache_entry(
    path: str | Path,
    mesh,
    landmarks: np.ndarray,
    subject_id: str,
    ear: str,
) -> Mapping[str, object]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = compute_vertex_geodesics(mesh, landmarks, ear)
    fingerprint = mesh_sha256(mesh)
    annotation_fingerprint = landmark_sha256(landmarks)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(GEODESIC_CACHE_SCHEMA_VERSION, dtype=np.int64),
            subject_id=np.asarray(subject_id),
            ear=np.asarray(ear),
            mesh_sha256=np.asarray(fingerprint),
            landmarks_sha256=np.asarray(annotation_fingerprint),
            **values,
        )
    return {
        "path": str(output),
        "sha256": file_sha256(output),
        "mesh_sha256": fingerprint,
        "landmarks_sha256": annotation_fingerprint,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "finite_fraction": float(np.isfinite(values["vertex_distances_mm"]).mean()),
        "maximum_source_projection_error_mm": float(
            np.max(values["source_projection_error_mm"])
        ),
    }


def load_geodesic_cache_entry(path: str | Path, mesh) -> dict:
    with np.load(path, allow_pickle=False) as data:
        schema = int(data["schema_version"])
        if schema != GEODESIC_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported geodesic cache schema {schema}; expected "
                f"{GEODESIC_CACHE_SCHEMA_VERSION}"
            )
        cached_hash = str(data["mesh_sha256"].item())
        actual_hash = mesh_sha256(mesh)
        if cached_hash != actual_hash:
            raise ValueError(
                "geodesic cache crop mesh does not match current calibration/centre"
            )
        result = {key: np.asarray(data[key]) for key in data.files}
    fields = np.asarray(result["vertex_distances_mm"], dtype=np.float32)
    if fields.shape != (85, len(mesh.vertices)):
        raise ValueError("geodesic cache has an invalid vertex-distance shape")
    return result


def sample_geodesic_distances_from_barycentric(
    mesh,
    face_indices: np.ndarray,
    barycentric: np.ndarray,
    cache: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate virtual-source graph distances at sampled triangle points."""
    indices = np.asarray(face_indices, dtype=np.int64)
    weights = np.asarray(barycentric, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(faces)):
        raise ValueError("sample face indices are invalid for the crop mesh")
    if weights.shape != (len(indices), 3) or not np.isfinite(weights).all():
        raise ValueError("sample barycentric coordinates must have shape (N, 3)")
    sample_faces = faces[indices]
    triangle_vertices = vertices[sample_faces]
    sample_xyz = np.sum(triangle_vertices * weights[:, :, None], axis=1)
    within_face = np.linalg.norm(
        sample_xyz[:, None, :] - triangle_vertices, axis=-1
    )
    vertex_fields = np.asarray(cache["vertex_distances_mm"], dtype=np.float64)
    candidates = vertex_fields[:, sample_faces] + within_face[None, :, :]
    distances = np.min(candidates, axis=-1)

    source_faces = np.asarray(cache["source_face_indices"], dtype=np.int64)
    source_points = np.asarray(cache["source_projected"], dtype=np.float64)
    for landmark_index in range(85):
        same_face = indices == source_faces[landmark_index]
        if np.any(same_face):
            distances[landmark_index, same_face] = np.linalg.norm(
                sample_xyz[same_face] - source_points[landmark_index], axis=-1
            )
    return distances.astype(np.float32)


def prepare_geodesic_cache(
    dataset,
    subject_ids: Sequence[str],
    center_predictions: Mapping[str, Sequence[float]],
    calibration: Mapping[str, object],
    output_dir: str | Path,
    folds_json: str | Path,
    predictions_json: str | Path,
    calibration_json: str | Path,
    outer_fold: int | str,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    id_to_index = {
        dataset.get_identifier(index): index for index in range(len(dataset))
    }
    entries = {}
    for subject_id in subject_ids:
        if subject_id not in id_to_index:
            raise ValueError(f"unknown geodesic-cache subject: {subject_id}")
        mesh, left, right = dataset[id_to_index[subject_id]]
        for ear, landmarks in zip(EAR_NAMES, (left, right)):
            key = prediction_key(subject_id, ear)
            if key not in center_predictions:
                raise ValueError(f"missing centre prediction for {key}")
            prepared = prepare_ear_geometry(
                mesh,
                landmarks,
                ear,
                center_predictions[key],
                calibration,
                num_points=1,
                seed=0,
            )
            entry_path = cache_path(root, subject_id, ear)
            cached = None
            if entry_path.is_file():
                try:
                    candidate = load_geodesic_cache_entry(
                        entry_path, prepared.crop_mesh
                    )
                    if (
                        str(candidate["subject_id"].item()) == subject_id
                        and str(candidate["ear"].item()) == ear
                        and str(candidate["landmarks_sha256"].item())
                        == landmark_sha256(landmarks)
                    ):
                        cached = candidate
                except (KeyError, OSError, ValueError):
                    cached = None
            if cached is None:
                entries[key] = save_geodesic_cache_entry(
                    entry_path,
                    prepared.crop_mesh,
                    landmarks,
                    subject_id,
                    ear,
                )
            else:
                entries[key] = {
                    "path": str(entry_path),
                    "sha256": file_sha256(entry_path),
                    "mesh_sha256": str(cached["mesh_sha256"].item()),
                    "landmarks_sha256": str(cached["landmarks_sha256"].item()),
                    "vertex_count": int(len(prepared.crop_mesh.vertices)),
                    "face_count": int(len(prepared.crop_mesh.faces)),
                    "finite_fraction": float(
                        np.isfinite(cached["vertex_distances_mm"]).mean()
                    ),
                    "maximum_source_projection_error_mm": float(
                        np.max(cached["source_projection_error_mm"])
                    ),
                }
            print(f"Prepared geodesic target {len(entries)}/{len(subject_ids) * 2}: {key}")

    manifest = {
        "schema_version": GEODESIC_CACHE_SCHEMA_VERSION,
        "component": "fold_mesh_geodesic_targets",
        "outer_fold": (
            "final" if str(outer_fold) == "final" else int(outer_fold)
        ),
        "subject_ids": list(subject_ids),
        "subject_count": len(subject_ids),
        "ear_count": len(entries),
        "folds_json_sha256": file_sha256(folds_json),
        "predictions_json_sha256": file_sha256(predictions_json),
        "calibration_json_sha256": file_sha256(calibration_json),
        "method": "triangle_virtual_source_mesh_edge_dijkstra",
        "entries": entries,
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def validate_geodesic_manifest(
    root: str | Path,
    outer_fold: int | str,
    subject_ids: Sequence[str],
    folds_json: str | Path,
    predictions_json: str | Path,
    calibration_json: str | Path,
) -> dict:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"missing geodesic cache manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    checks = {
        "outer_fold": (
            "final" if str(outer_fold) == "final" else int(outer_fold)
        ),
        "folds_json_sha256": file_sha256(folds_json),
        "predictions_json_sha256": file_sha256(predictions_json),
        "calibration_json_sha256": file_sha256(calibration_json),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(f"geodesic cache manifest has wrong {key}")
    if list(manifest.get("subject_ids", [])) != list(subject_ids):
        raise ValueError("geodesic cache subject membership/order does not match fold")
    expected_keys = {
        prediction_key(subject_id, ear)
        for subject_id in subject_ids
        for ear in EAR_NAMES
    }
    if set(manifest.get("entries", {})) != expected_keys:
        raise ValueError("geodesic cache does not cover exactly the requested fold ears")
    missing = [
        key for key in sorted(expected_keys)
        if not cache_path(root, *key.split(":")).is_file()
    ]
    if missing:
        raise ValueError(f"geodesic cache files are missing: {missing[:5]}")
    corrupt = []
    for key in sorted(expected_keys):
        path = cache_path(root, *key.split(":"))
        if file_sha256(path) != manifest["entries"][key].get("sha256"):
            corrupt.append(key)
    if corrupt:
        raise ValueError(
            f"geodesic cache file checksums do not match manifest: {corrupt[:5]}"
        )
    return manifest
