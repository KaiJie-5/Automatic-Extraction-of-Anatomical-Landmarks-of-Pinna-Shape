"""Deterministic intrinsic/differential features for sampled ear surfaces.

The established pipeline supplies crop-local XYZ followed by canonicalized
unit normals.  This module augments those six channels without changing the
sampled candidates or their ordering.  All differential quantities are
computed from the same deterministic sample and are bounded to keep the input
scale stable across meshes with different triangulation densities.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


BASE_POINT_CHANNELS = 6
SURFACE_GEOMETRY_SCHEMA_VERSION = 1
DEFAULT_RADII_MM = (1.0, 2.0, 4.0)
DEFAULT_NEIGHBOURS = 64
DEFAULT_CURVATURE_RADIUS_MM = 2.0


def feature_names(radii_mm: Sequence[float] = DEFAULT_RADII_MM) -> tuple[str, ...]:
    radii = tuple(float(value) for value in radii_mm)
    return tuple(f"normal_variation_r{value:g}mm" for value in radii) + (
        "bounded_abs_mean_curvature",
        "bounded_signed_gaussian_curvature",
        "abs_shape_index",
        "bounded_curvedness",
        "crop_centre_distance",
    )


def make_surface_geometry_config(
    enabled: bool = False,
    radii_mm: Sequence[float] = DEFAULT_RADII_MM,
    neighbours: int = DEFAULT_NEIGHBOURS,
    curvature_radius_mm: float = DEFAULT_CURVATURE_RADIUS_MM,
) -> dict:
    """Build and validate serializable preprocessing metadata."""

    radii = tuple(float(value) for value in radii_mm)
    if len(radii) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in radii):
        raise ValueError("surface geometry requires three positive finite radii")
    if tuple(sorted(radii)) != radii or len(set(radii)) != len(radii):
        raise ValueError("surface geometry radii must be strictly increasing")
    neighbours = int(neighbours)
    if neighbours not in {32, 64, 128}:
        raise ValueError("surface geometry neighbours must be 32, 64, or 128")
    curvature_radius_mm = float(curvature_radius_mm)
    if not math.isfinite(curvature_radius_mm) or curvature_radius_mm <= 0.0:
        raise ValueError("surface curvature radius must be positive and finite")
    if curvature_radius_mm > radii[-1]:
        raise ValueError("surface curvature radius cannot exceed the largest geometry radius")
    names = feature_names(radii)
    return {
        "enabled": bool(enabled),
        "schema_version": SURFACE_GEOMETRY_SCHEMA_VERSION,
        "method": "sample_knn_normal_shape_operator",
        "radii_mm": list(radii),
        "neighbours": neighbours,
        "curvature_radius_mm": curvature_radius_mm,
        "normal_variation_orientation": "oriented_input_normals",
        "curvature_normal_orientation": "align_to_query_normal",
        "feature_names": list(names),
        "base_channels": BASE_POINT_CHANNELS,
        "output_channels": BASE_POINT_CHANNELS + len(names) if enabled else BASE_POINT_CHANNELS,
    }


def validate_surface_geometry_config(config: Mapping[str, object] | None) -> dict:
    """Return canonical metadata and reject incomplete checkpoint settings."""

    if not config:
        return make_surface_geometry_config(enabled=False)
    required = {
        "enabled",
        "schema_version",
        "method",
        "radii_mm",
        "neighbours",
        "curvature_radius_mm",
        "normal_variation_orientation",
        "curvature_normal_orientation",
        "feature_names",
        "base_channels",
        "output_channels",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"incomplete surface geometry configuration; missing: {missing}")
    if int(config["schema_version"]) != SURFACE_GEOMETRY_SCHEMA_VERSION:
        raise ValueError("unsupported surface geometry schema version")
    if str(config["method"]) != "sample_knn_normal_shape_operator":
        raise ValueError("unsupported surface geometry feature method")
    canonical = make_surface_geometry_config(
        enabled=bool(config["enabled"]),
        radii_mm=config["radii_mm"],
        neighbours=int(config["neighbours"]),
        curvature_radius_mm=float(config["curvature_radius_mm"]),
    )
    if list(config["feature_names"]) != canonical["feature_names"]:
        raise ValueError("surface geometry feature names/order do not match the schema")
    for key in (
        "normal_variation_orientation",
        "curvature_normal_orientation",
    ):
        if str(config[key]) != canonical[key]:
            raise ValueError(f"surface geometry {key} does not match the schema")
    if int(config["base_channels"]) != canonical["base_channels"]:
        raise ValueError("surface geometry base channel count does not match the schema")
    if int(config["output_channels"]) != canonical["output_channels"]:
        raise ValueError("surface geometry output channel count does not match the schema")
    return canonical


def surface_geometry_from_model_config(model_config: Mapping[str, object]) -> dict:
    return validate_surface_geometry_config(model_config.get("surface_geometry"))


def _unit_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct deterministic orthonormal tangent frames."""

    axes = np.eye(3, dtype=np.float64)
    least_aligned = np.argmin(np.abs(normals), axis=1)
    reference = axes[least_aligned]
    tangent_one = _unit_vectors(np.cross(normals, reference))
    tangent_two = _unit_vectors(np.cross(normals, tangent_one))
    return tangent_one, tangent_two


def _bounded_curvature_features(
    xyz: np.ndarray,
    normals: np.ndarray,
    neighbour_xyz: np.ndarray,
    aligned_neighbour_normals: np.ndarray,
    distances: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Estimate a local two-dimensional normal shape operator.

    The least-squares relation ``-dn = S dx`` is fitted in a deterministic
    tangent frame.  Its eigenvalues approximate the principal curvatures in
    normalized-coordinate units.  Products with the fitting radius make the
    exported channels dimensionless before a tanh bound is applied.
    """

    count = xyz.shape[0]
    result = np.zeros((count, 4), dtype=np.float64)
    if count == 0:
        return result.astype(np.float32)
    tangent_one, tangent_two = _tangent_frames(normals)
    relative = neighbour_xyz - xyz[:, None, :]
    normal_delta = aligned_neighbour_normals - normals[:, None, :]
    tangent_offsets = np.stack(
        [
            np.einsum("nki,ni->nk", relative, tangent_one),
            np.einsum("nki,ni->nk", relative, tangent_two),
        ],
        axis=-1,
    )
    tangent_normal_delta = -np.stack(
        [
            np.einsum("nki,ni->nk", normal_delta, tangent_one),
            np.einsum("nki,ni->nk", normal_delta, tangent_two),
        ],
        axis=-1,
    )
    valid = (distances > 1e-8) & (distances <= float(radius))
    valid_count = valid.sum(axis=1)
    weights = np.exp(-0.5 * np.square(distances / max(float(radius), 1e-8))) * valid
    design = tangent_offsets
    normal_response = tangent_normal_delta
    normal_matrix = np.einsum("nki,nk,nkj->nij", design, weights, design)
    response_matrix = np.einsum(
        "nki,nk,nkj->nij", design, weights, normal_response
    )
    regularizer = max(float(radius) ** 2 * 1e-4, 1e-8)
    normal_matrix[:, 0, 0] += regularizer
    normal_matrix[:, 1, 1] += regularizer
    try:
        shape_operator = np.linalg.solve(normal_matrix, response_matrix)
    except np.linalg.LinAlgError:
        shape_operator = np.matmul(np.linalg.pinv(normal_matrix), response_matrix)
    shape_operator = 0.5 * (
        shape_operator + np.swapaxes(shape_operator, 1, 2)
    )
    principal = np.linalg.eigvalsh(shape_operator)
    valid_rows = valid_count >= 5
    principal[~valid_rows] = 0.0
    k_min = principal[:, 0]
    k_max = principal[:, 1]
    mean = 0.5 * (k_min + k_max)
    gaussian = k_min * k_max
    curvedness = np.sqrt(0.5 * (np.square(k_min) + np.square(k_max)))
    shape_index = np.abs(
        (2.0 / np.pi) * np.arctan2(k_max + k_min, k_max - k_min)
    )
    shape_index[(np.abs(k_min) + np.abs(k_max)) < 1e-10] = 0.0
    result[:, 0] = np.tanh(float(radius) * np.abs(mean))
    # Gaussian curvature is independent of normal orientation, and its sign
    # separates elliptic regions from saddles.  Preserve that information.
    result[:, 1] = np.tanh(float(radius) ** 2 * gaussian)
    result[:, 2] = np.clip(shape_index, 0.0, 1.0)
    result[:, 3] = np.tanh(float(radius) * curvedness)
    result[~valid_rows] = 0.0
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0).astype(
        np.float32
    )


def append_surface_geometry_features(
    local_point_features: np.ndarray,
    local_scale_mm: float,
    config: Mapping[str, object] | None,
) -> np.ndarray:
    """Append bounded differential features without changing point order."""

    canonical = validate_surface_geometry_config(config)
    values = np.asarray(local_point_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != BASE_POINT_CHANNELS:
        raise ValueError("surface geometry expects local point features with shape (N, 6)")
    if not np.isfinite(values).all():
        raise ValueError("surface geometry input must be finite")
    if not canonical["enabled"]:
        return values.copy()
    local_scale_mm = float(local_scale_mm)
    if not math.isfinite(local_scale_mm) or local_scale_mm <= 0.0:
        raise ValueError("surface geometry local scale must be positive and finite")
    if len(values) < 2:
        raise ValueError("surface geometry requires at least two sampled points")

    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise ImportError(
            "surface geometry features require scipy; install the repository requirements"
        ) from error

    xyz = values[:, :3].astype(np.float64)
    normals = _unit_vectors(values[:, 3:6].astype(np.float64))
    neighbour_count = min(int(canonical["neighbours"]), len(values))
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz, k=neighbour_count, workers=1)
    if neighbour_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    neighbour_xyz = xyz[indices]
    neighbour_normals = normals[indices]
    normal_dot = np.einsum("nki,ni->nk", neighbour_normals, normals)
    orientation = np.where(normal_dot < 0.0, -1.0, 1.0)
    aligned_normals = neighbour_normals * orientation[..., None]

    variations = []
    for radius_mm in canonical["radii_mm"]:
        radius = float(radius_mm) / local_scale_mm
        mask = distances <= radius
        weights = np.exp(-0.5 * np.square(distances / max(radius, 1e-8))) * mask
        denominator = weights.sum(axis=1, keepdims=True)
        average = np.einsum("nk,nki->ni", weights, neighbour_normals) / np.maximum(
            denominator, 1e-8
        )
        variation = 1.0 - np.linalg.norm(average, axis=1)
        variations.append(np.clip(variation, 0.0, 1.0))

    curvature_radius = float(canonical["curvature_radius_mm"]) / local_scale_mm
    curvature = _bounded_curvature_features(
        xyz,
        normals,
        neighbour_xyz,
        aligned_normals,
        distances,
        curvature_radius,
    )
    radial_distance = np.linalg.norm(xyz, axis=1, keepdims=True)
    extra = np.concatenate(
        [np.stack(variations, axis=1), curvature, radial_distance], axis=1
    ).astype(np.float32)
    output = np.concatenate([values, extra], axis=1).astype(np.float32)
    if output.shape[1] != int(canonical["output_channels"]):
        raise RuntimeError("surface geometry feature count does not match its schema")
    if not np.isfinite(output).all():
        raise RuntimeError("surface geometry preprocessing produced non-finite values")
    return output
