"""Continuous anatomical-curve targets shared by training and inference.

The challenge landmarks form four open, ordered contours.  A normalized
arc-length coordinate is therefore well defined independently for every
contour.  Training examples use their own annotation-derived fractions while
inference uses the component-wise training-fold median serialized in the
checkpoint.  This keeps the representation fold-safe and deterministic.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


CONTOUR_NAMES = (
    "outer_helix",
    "concha",
    "inner_helix",
    "superior_antihelix",
)
CONTOUR_RANGES = ((0, 25), (25, 55), (55, 75), (75, 85))
CONTOUR_ANCHORS = (
    (0, 6, 22, 24),
    (25, 33, 42, 46, 50, 54),
    (55, 64, 74),
    (75, 84),
)


def contour_anchor_manifest() -> dict[str, list[int]]:
    """Return the fixed semantic waypoints used by connected path decoding."""

    return {
        name: [int(index) for index in anchors]
        for name, anchors in zip(CONTOUR_NAMES, CONTOUR_ANCHORS)
    }


def landmark_arc_fractions(landmarks: np.ndarray) -> np.ndarray:
    """Return per-landmark normalized chord arc lengths for four open curves.

    Consecutive annotations are sufficiently dense that their cumulative chord
    length is a stable proxy for surface arc length.  Fractions are invariant
    to rigid transforms, mirroring, translation, and uniform scaling.
    """

    values = np.asarray(landmarks, dtype=np.float64)
    if values.shape != (85, 3) or not np.isfinite(values).all():
        raise ValueError("curve landmarks must be finite with shape (85, 3)")
    result = np.empty(85, dtype=np.float32)
    for start, end in CONTOUR_RANGES:
        curve = values[start:end]
        intervals = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(intervals)])
        total = float(cumulative[-1])
        if not np.isfinite(total) or total <= 1e-8:
            raise ValueError("every anatomical contour must have positive length")
        result[start:end] = (cumulative / total).astype(np.float32)
    return result


def validate_landmark_arc_fractions(fractions: Sequence[float]) -> np.ndarray:
    """Validate one serialized 85-value, strictly ordered fraction vector."""

    values = np.asarray(fractions, dtype=np.float64)
    if values.shape != (85,) or not np.isfinite(values).all():
        raise ValueError("curve landmark fractions must contain 85 finite values")
    for start, end in CONTOUR_RANGES:
        contour = values[start:end]
        if not np.isclose(contour[0], 0.0, atol=1e-6):
            raise ValueError("every curve fraction sequence must start at zero")
        if not np.isclose(contour[-1], 1.0, atol=1e-6):
            raise ValueError("every curve fraction sequence must end at one")
        if np.any(np.diff(contour) <= 0.0):
            raise ValueError("curve landmark fractions must be strictly increasing")
    return values.astype(np.float32)


def median_landmark_arc_fractions(
    landmark_sets: Iterable[np.ndarray],
) -> np.ndarray:
    """Calculate a leakage-safe inference template from training ears only."""

    rows = [landmark_arc_fractions(values) for values in landmark_sets]
    if not rows:
        raise ValueError("at least one training ear is required for curve fractions")
    median = np.median(np.stack(rows, axis=0), axis=0).astype(np.float32)
    # Median preserves order in theory; setting the exact endpoints avoids
    # serialization round-off and validates that no malformed input slipped in.
    for start, end in CONTOUR_RANGES:
        median[start] = 0.0
        median[end - 1] = 1.0
    return validate_landmark_arc_fractions(median)


def contour_ids() -> np.ndarray:
    """Map the official 85 landmark indices to their contour index."""

    result = np.empty(85, dtype=np.int64)
    for contour, (start, end) in enumerate(CONTOUR_RANGES):
        result[start:end] = contour
    return result


def _mesh_edges_with_lengths(mesh, vertices: np.ndarray):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    # Plane slicing can leave coincident boundary vertices with distinct
    # indices.  Explicit tiny edges preserve the same connectivity convention
    # as the geodesic-target cache.
    _, groups = np.unique(vertices, axis=0, return_inverse=True)
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    duplicate_edges = []
    for group in np.split(order, boundaries):
        if len(group) > 1:
            duplicate_edges.append(
                np.column_stack(
                    [
                        np.full(len(group) - 1, group[0], dtype=np.int64),
                        group[1:],
                    ]
                )
            )
    if duplicate_edges:
        edges = np.concatenate([edges, *duplicate_edges], axis=0)
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
    )
    valid = np.isfinite(lengths) & (edges[:, 0] != edges[:, 1])
    return edges[valid], np.maximum(lengths[valid], 1e-12)


def _interpolate_sample_fields(
    sample_xyz: np.ndarray,
    vertex_xyz: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    count = min(4, len(sample_xyz))
    distances, indices = cKDTree(sample_xyz).query(vertex_xyz, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-8)
    exact = distances <= 1e-8
    has_exact = exact.any(axis=1)
    if np.any(has_exact):
        weights[has_exact] = exact[has_exact].astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(values[:, indices] * weights[None, :, :], axis=-1)


def decode_connected_curve_paths(
    mesh,
    ear: str,
    transform,
    sample_local_xyz: np.ndarray,
    curve_logits: np.ndarray,
    curve_arc_coordinates: np.ndarray,
    initial_landmarks_local: np.ndarray,
    landmark_fractions: Sequence[float],
    field_strength: float = 4.0,
    backtrack_weight: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """Decode ordered, anchor-routed paths on the exact cropped mesh.

    The neural curve field supplies the data term.  Mesh edges enforce surface
    connectivity, and decreasing predicted arc coordinates receive an
    asymmetric cost. Each anatomical contour is split at its known semantic
    section anchors so a closed or folded contour cannot take the geometrically
    short route directly from its first landmark to its last. Outputs inside
    each section are interpolated using section-local training-fold fractions.
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    from .canonical import canonicalize_xyz

    samples = np.asarray(sample_local_xyz, dtype=np.float64)
    logits = np.asarray(curve_logits, dtype=np.float64)
    arc = np.asarray(curve_arc_coordinates, dtype=np.float64)
    initial = np.asarray(initial_landmarks_local, dtype=np.float64)
    fractions = validate_landmark_arc_fractions(landmark_fractions).astype(
        np.float64
    )
    field_strength = float(field_strength)
    backtrack_weight = float(backtrack_weight)
    if samples.ndim != 2 or samples.shape[1] != 3 or not len(samples):
        raise ValueError("curve path samples must have shape (N, 3)")
    if logits.shape != (4, len(samples)) or arc.shape != logits.shape:
        raise ValueError("curve path fields must have shape (4, N)")
    if initial.shape != (85, 3):
        raise ValueError("initial curve landmarks must have shape (85, 3)")
    if not (
        np.isfinite(samples).all()
        and np.isfinite(logits).all()
        and np.isfinite(arc).all()
        and np.isfinite(initial).all()
    ):
        raise ValueError("curve path inputs must be finite")
    if not np.isfinite(field_strength) or field_strength < 0.0:
        raise ValueError("curve path field strength must be non-negative")
    if not np.isfinite(backtrack_weight) or backtrack_weight < 0.0:
        raise ValueError("curve path backtrack weight must be non-negative")

    world_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if world_vertices.ndim != 2 or world_vertices.shape[1] != 3:
        raise ValueError("curve path mesh has invalid vertices")
    canonical_vertices = canonicalize_xyz(world_vertices, ear)
    vertex_local = transform.normalize_xyz(canonical_vertices).astype(np.float64)
    vertex_logits = _interpolate_sample_fields(samples, vertex_local, logits)
    vertex_arc = np.clip(
        _interpolate_sample_fields(samples, vertex_local, arc), 0.0, 1.0
    )
    edges, lengths = _mesh_edges_with_lengths(mesh, vertex_local)
    if not len(edges):
        raise ValueError("curve path mesh has no valid edges")

    from scipy.spatial import cKDTree

    vertex_tree = cKDTree(vertex_local)
    output = initial.copy()
    diagnostics = {}
    for contour, ((start, end), anchors, name) in enumerate(
        zip(CONTOUR_RANGES, CONTOUR_ANCHORS, CONTOUR_NAMES)
    ):
        if anchors[0] != start or anchors[-1] != end - 1:
            raise RuntimeError("curve anchor definitions do not match contour ranges")
        anchor_vertices = [
            int(vertex_tree.query(initial[index], k=1)[1])
            for index in anchors
        ]
        record = {
            "anchor_indices": [int(index) for index in anchors],
            "anchor_vertices": anchor_vertices,
            "section_count": int(len(anchors) - 1),
            "successful_section_count": 0,
            "sections": [],
            "fallback": False,
        }

        field = vertex_logits[contour]
        low, high = np.percentile(field, [5.0, 95.0])
        span = max(float(high - low), 1e-8)
        quality = np.clip((field - low) / span, 0.0, 1.0)
        node_penalty = 1.0 + field_strength * (1.0 - quality)
        left = edges[:, 0]
        right = edges[:, 1]
        base = lengths * 0.5 * (node_penalty[left] + node_penalty[right])
        left_to_right = base * (
            1.0
            + backtrack_weight
            * np.maximum(vertex_arc[contour, left] - vertex_arc[contour, right], 0.0)
        )
        right_to_left = base * (
            1.0
            + backtrack_weight
            * np.maximum(vertex_arc[contour, right] - vertex_arc[contour, left], 0.0)
        )
        rows = np.concatenate([left, right])
        columns = np.concatenate([right, left])
        costs = np.concatenate([left_to_right, right_to_left])
        graph = coo_matrix(
            (costs, (rows, columns)),
            shape=(len(vertex_local), len(vertex_local)),
        ).tocsr()
        distance, predecessors = dijkstra(
            graph,
            directed=True,
            indices=np.asarray(anchor_vertices[:-1], dtype=np.int64),
            return_predecessors=True,
        )
        distance = np.atleast_2d(np.asarray(distance, dtype=np.float64))
        predecessors = np.atleast_2d(np.asarray(predecessors))
        total_length = 0.0
        total_cost = 0.0
        total_vertices = 0
        for section_index, (section_start, section_end) in enumerate(
            zip(anchors[:-1], anchors[1:])
        ):
            start_vertex = anchor_vertices[section_index]
            end_vertex = anchor_vertices[section_index + 1]
            section_record = {
                "start_index": int(section_start),
                "end_index": int(section_end),
                "start_vertex": int(start_vertex),
                "end_vertex": int(end_vertex),
                "fallback": False,
            }
            if start_vertex == end_vertex:
                section_record.update(
                    {"fallback": True, "reason": "identical_anchor_vertices"}
                )
                record["fallback"] = True
                record["sections"].append(section_record)
                continue
            if not np.isfinite(distance[section_index, end_vertex]):
                section_record.update(
                    {"fallback": True, "reason": "disconnected_anchors"}
                )
                record["fallback"] = True
                record["sections"].append(section_record)
                continue

            reversed_path = [end_vertex]
            current = end_vertex
            while (
                current != start_vertex
                and len(reversed_path) <= len(vertex_local)
            ):
                current = int(predecessors[section_index, current])
                if current < 0:
                    break
                reversed_path.append(current)
            if not reversed_path or reversed_path[-1] != start_vertex:
                section_record.update(
                    {"fallback": True, "reason": "invalid_predecessors"}
                )
                record["fallback"] = True
                record["sections"].append(section_record)
                continue

            path_indices = np.asarray(reversed_path[::-1], dtype=np.int64)
            path = vertex_local[path_indices]
            segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
            section_length = float(cumulative[-1])
            if section_length <= 1e-8:
                section_record.update(
                    {"fallback": True, "reason": "zero_length_path"}
                )
                record["fallback"] = True
                record["sections"].append(section_record)
                continue

            fraction_start = float(fractions[section_start])
            fraction_end = float(fractions[section_end])
            fraction_span = fraction_end - fraction_start
            if fraction_span <= 1e-8:
                raise RuntimeError("curve section fractions are not increasing")
            section_fractions = (
                fractions[section_start : section_end + 1] - fraction_start
            ) / fraction_span
            targets = section_fractions * section_length
            segment = np.searchsorted(cumulative, targets, side="right") - 1
            segment = np.clip(segment, 0, len(path) - 2)
            denominator = np.maximum(
                cumulative[segment + 1] - cumulative[segment], 1e-12
            )
            alpha = (targets - cumulative[segment]) / denominator
            output[section_start : section_end + 1] = (
                path[segment] * (1.0 - alpha[:, None])
                + path[segment + 1] * alpha[:, None]
            )
            weighted_cost = float(distance[section_index, end_vertex])
            section_record.update(
                {
                    "vertex_count": int(len(path_indices)),
                    "path_length_local": section_length,
                    "weighted_path_cost": weighted_cost,
                }
            )
            record["successful_section_count"] += 1
            record["sections"].append(section_record)
            total_length += section_length
            total_cost += weighted_cost
            total_vertices += int(len(path_indices))
        record.update(
            {
                "vertex_count": total_vertices,
                "path_length_local": total_length,
                "weighted_path_cost": total_cost,
            }
        )
        if record["fallback"]:
            record["reason"] = "one_or_more_sections_failed"
        diagnostics[name] = record
    if not np.isfinite(output).all():
        raise RuntimeError("connected curve decoder produced non-finite landmarks")
    return output.astype(np.float32), diagnostics
