"""Proposal-aligned landmark objectives and experiment promotion."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from .curve import CONTOUR_RANGES


ANCHOR_INDICES = (0, 6, 22, 24, 25, 33, 42, 46, 50, 54, 55, 64, 74, 75, 84)
SPACING_SECTIONS = ((0, 6), (6, 22), (22, 24), (25, 33), (33, 42), (42, 46), (46, 50), (50, 54), (55, 64), (64, 74), (75, 84))


def mean_distance_mm(prediction: torch.Tensor, target: torch.Tensor, scale_mm) -> torch.Tensor:
    scale = torch.as_tensor(scale_mm, dtype=prediction.dtype, device=prediction.device)
    while scale.ndim < prediction.ndim - 1:
        scale = scale.unsqueeze(-1)
    return (torch.linalg.norm(prediction - target, dim=-1) * scale).mean()


def anchor_distance_mm(prediction: torch.Tensor, target: torch.Tensor, scale_mm) -> torch.Tensor:
    indices = torch.as_tensor(ANCHOR_INDICES, device=prediction.device)
    return mean_distance_mm(prediction.index_select(1, indices), target.index_select(1, indices), scale_mm)


def spacing_uniformity_mm(prediction: torch.Tensor, scale_mm) -> torch.Tensor:
    terms = []
    for start, end in SPACING_SECTIONS:
        spacing = torch.linalg.norm(
            prediction[:, start + 1 : end + 1] - prediction[:, start:end], dim=-1
        )
        terms.append(torch.abs(spacing - spacing.mean(dim=1, keepdim=True)).mean())
    scale = torch.as_tensor(scale_mm, dtype=prediction.dtype, device=prediction.device)
    return torch.stack(terms).mean() * scale.mean()


def predicted_to_surface_mm(
    prediction: torch.Tensor,
    dense_surface: torch.Tensor,
    scale_mm,
    surface_chunk: int = 4096,
) -> torch.Tensor:
    minima = []
    for start in range(0, dense_surface.shape[1], surface_chunk):
        distances = torch.cdist(prediction, dense_surface[:, start : start + surface_chunk, :])
        minima.append(distances.amin(dim=-1))
    minimum = torch.stack(minima, dim=0).amin(dim=0)
    scale = torch.as_tensor(scale_mm, dtype=prediction.dtype, device=prediction.device)
    while scale.ndim < minimum.ndim:
        scale = scale.unsqueeze(-1)
    return (minimum * scale).mean()


def surface_heatmap_kl(
    logits: torch.Tensor,
    surface_points: torch.Tensor,
    target: torch.Tensor,
    scale_mm,
    sigma_mm: float,
) -> torch.Tensor:
    """KL divergence to Gaussian landmark heatmaps on sampled surface points.

    The target distribution is constructed in original millimetres even though
    model coordinates are crop-local and normalized.  KL, rather than plain
    cross entropy, removes the target entropy constant so a perfect heatmap has
    zero auxiliary loss and its weight is easier to interpret beside MD in mm.
    """
    if logits.ndim != 3 or logits.shape[1] != 85:
        raise ValueError("heatmap logits must have shape (B, 85, N)")
    if surface_points.ndim != 3 or surface_points.shape[-1] != 3:
        raise ValueError("surface heatmap points must have shape (B, N, 3)")
    if target.ndim != 3 or target.shape[1:] != (85, 3):
        raise ValueError("surface heatmap targets must have shape (B, 85, 3)")
    if logits.shape[0] != surface_points.shape[0] or logits.shape[0] != target.shape[0]:
        raise ValueError("surface heatmap batch dimensions must match")
    if logits.shape[2] != surface_points.shape[1]:
        raise ValueError("heatmap logits and candidate points must have matching N")
    sigma_mm = float(sigma_mm)
    if not math.isfinite(sigma_mm) or sigma_mm <= 0.0:
        raise ValueError("surface heatmap sigma must be positive and finite")

    points = surface_points.float()
    targets = target.float()
    scale = torch.as_tensor(scale_mm, dtype=torch.float32, device=points.device)
    if scale.ndim == 0:
        scale = scale.expand(points.shape[0])
    scale = scale.reshape(points.shape[0], -1)
    if scale.shape[1] != 1:
        raise ValueError("surface heatmap scale must contain one value per batch item")
    delta_mm = (
        points[:, None, :, :] - targets[:, :, None, :]
    ) * scale[:, None, None, :]
    squared_mm = torch.sum(delta_mm * delta_mm, dim=-1)
    target_logits = -squared_mm / (2.0 * sigma_mm * sigma_mm)
    target_probabilities = torch.softmax(target_logits, dim=-1)
    target_log_probabilities = torch.log_softmax(target_logits, dim=-1)
    predicted_log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    finite_target_logs = torch.where(
        target_probabilities > 0.0,
        target_log_probabilities,
        torch.zeros_like(target_log_probabilities),
    )
    terms = target_probabilities * (
        finite_target_logs - predicted_log_probabilities
    )
    divergence = terms.sum(dim=-1).mean()
    # KL is mathematically non-negative, but an almost perfect distribution can
    # accumulate a tiny negative value (around 1e-21) in finite precision.
    # Clamping restores the invariant without affecting any meaningful loss.
    return divergence.clamp_min(0.0)


def surface_geodesic_heatmap_kl(
    logits: torch.Tensor,
    geodesic_distances_mm: torch.Tensor,
    sigma_mm: float,
) -> torch.Tensor:
    """KL to a Gaussian defined by mesh-geodesic, not chord, distance."""
    if logits.ndim != 3 or logits.shape[1] != 85:
        raise ValueError("heatmap logits must have shape (B, 85, N)")
    if geodesic_distances_mm.shape != logits.shape:
        raise ValueError(
            "geodesic heatmap distances must match logits with shape (B, 85, N)"
        )
    sigma_mm = float(sigma_mm)
    if not math.isfinite(sigma_mm) or sigma_mm <= 0.0:
        raise ValueError("surface heatmap sigma must be positive and finite")
    distances = geodesic_distances_mm.float()
    if torch.isnan(distances).any() or (distances < 0.0).any():
        raise ValueError("geodesic distances must be non-negative and not NaN")
    if not torch.isfinite(distances).any(dim=-1).all():
        raise ValueError("every landmark requires at least one finite geodesic candidate")
    target_logits = -(distances.square()) / (2.0 * sigma_mm * sigma_mm)
    target_probabilities = torch.softmax(target_logits, dim=-1)
    target_log_probabilities = torch.log_softmax(target_logits, dim=-1)
    predicted_log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    finite_target_logs = torch.where(
        target_probabilities > 0.0,
        target_log_probabilities,
        torch.zeros_like(target_log_probabilities),
    )
    terms = target_probabilities * (
        finite_target_logs - predicted_log_probabilities
    )
    divergence = terms.sum(dim=-1).mean()
    return divergence.clamp_min(0.0)


def surface_vote_offset_loss(
    vote_offsets: torch.Tensor,
    surface_points: torch.Tensor,
    target: torch.Tensor,
    scale_mm,
    geodesic_distances_mm: torch.Tensor,
    radius_mm: float,
    sigma_mm: float,
) -> torch.Tensor:
    """Gaussian-weighted candidate-to-landmark voting error in millimetres."""
    expected = (*geodesic_distances_mm.shape, 3)
    if vote_offsets.shape != expected:
        raise ValueError(f"surface vote offsets must have shape {expected}")
    if surface_points.ndim != 3 or surface_points.shape[-1] != 3:
        raise ValueError("surface vote points must have shape (B, N, 3)")
    if target.ndim != 3 or target.shape[1:] != (85, 3):
        raise ValueError("surface vote targets must have shape (B, 85, 3)")
    radius_mm = float(radius_mm)
    sigma_mm = float(sigma_mm)
    if not math.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ValueError("surface vote radius must be positive and finite")
    if not math.isfinite(sigma_mm) or sigma_mm <= 0.0:
        raise ValueError("surface vote sigma must be positive and finite")

    scale = torch.as_tensor(scale_mm, dtype=torch.float32, device=target.device)
    if scale.ndim == 0:
        scale = scale.expand(target.shape[0])
    scale = scale.reshape(target.shape[0], -1)
    if scale.shape[1] != 1:
        raise ValueError("surface vote scale must contain one value per batch item")
    target_offsets = target.float()[:, :, None, :] - surface_points.float()[:, None, :, :]
    error_mm = torch.linalg.norm(
        (vote_offsets.float() - target_offsets) * scale[:, None, None, :],
        dim=-1,
    )
    distances = geodesic_distances_mm.float()
    mask = torch.isfinite(distances) & (distances <= radius_mm)
    weights = torch.where(
        mask,
        torch.exp(-distances.square() / (2.0 * sigma_mm * sigma_mm)),
        torch.zeros_like(distances),
    )
    denominator = weights.sum(dim=-1)
    if not (denominator > 0.0).all():
        raise ValueError(
            "surface vote radius contains no sampled candidate for at least one landmark"
        )
    per_landmark = (weights * error_mm).sum(dim=-1) / denominator
    return per_landmark.mean()


def surface_curve_field_kl(
    curve_logits: torch.Tensor,
    geodesic_distances_mm: torch.Tensor,
    sigma_mm: float,
) -> torch.Tensor:
    """KL supervision for four dense anatomical-contour surface fields."""

    if curve_logits.ndim != 3 or curve_logits.shape[1] != len(CONTOUR_RANGES):
        raise ValueError("curve logits must have shape (B, 4, N)")
    if (
        geodesic_distances_mm.ndim != 3
        or geodesic_distances_mm.shape[1] != 85
        or geodesic_distances_mm.shape[0] != curve_logits.shape[0]
        or geodesic_distances_mm.shape[2] != curve_logits.shape[2]
    ):
        raise ValueError("curve fields require geodesic distances with shape (B, 85, N)")
    sigma_mm = float(sigma_mm)
    if not math.isfinite(sigma_mm) or sigma_mm <= 0.0:
        raise ValueError("curve field sigma must be positive and finite")

    distances = geodesic_distances_mm.float()
    contour_distances = torch.stack(
        [distances[:, start:end].amin(dim=1) for start, end in CONTOUR_RANGES],
        dim=1,
    )
    if not torch.isfinite(contour_distances).any(dim=-1).all():
        raise ValueError("every contour requires at least one finite surface candidate")
    target_logits = -contour_distances.square() / (2.0 * sigma_mm * sigma_mm)
    target_probabilities = torch.softmax(target_logits, dim=-1)
    target_logs = torch.log_softmax(target_logits, dim=-1)
    predicted_logs = torch.log_softmax(curve_logits.float(), dim=-1)
    finite_target_logs = torch.where(
        target_probabilities > 0.0,
        target_logs,
        torch.zeros_like(target_logs),
    )
    divergence = (
        target_probabilities * (finite_target_logs - predicted_logs)
    ).sum(dim=-1).mean()
    return divergence.clamp_min(0.0)


def surface_curve_arc_loss(
    arc_coordinates: torch.Tensor,
    geodesic_distances_mm: torch.Tensor,
    landmark_fractions: torch.Tensor,
    sigma_mm: float,
    radius_mm: float,
) -> torch.Tensor:
    """Regress normalized position along each curve near its annotated trace.

    A sample's target coordinate is the Gaussian-weighted interpolation of the
    arc fractions of nearby ordered landmarks.  Only samples inside the
    geodesic supervision radius contribute, preventing unrelated surface sheets
    from corrupting the intrinsic coordinate.
    """

    if arc_coordinates.ndim != 3 or arc_coordinates.shape[1] != len(CONTOUR_RANGES):
        raise ValueError("curve arc coordinates must have shape (B, 4, N)")
    if (
        geodesic_distances_mm.ndim != 3
        or geodesic_distances_mm.shape[1] != 85
        or geodesic_distances_mm.shape[0] != arc_coordinates.shape[0]
        or geodesic_distances_mm.shape[2] != arc_coordinates.shape[2]
    ):
        raise ValueError("curve arc loss requires geodesic shape (B, 85, N)")
    if landmark_fractions.shape != geodesic_distances_mm.shape[:2]:
        raise ValueError("curve landmark fractions must have shape (B, 85)")
    sigma_mm = float(sigma_mm)
    radius_mm = float(radius_mm)
    if not math.isfinite(sigma_mm) or sigma_mm <= 0.0:
        raise ValueError("curve arc sigma must be positive and finite")
    if not math.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ValueError("curve arc radius must be positive and finite")

    distances = geodesic_distances_mm.float()
    fractions = landmark_fractions.float()
    weighted_error = arc_coordinates.new_zeros((), dtype=torch.float32)
    total_weight = arc_coordinates.new_zeros((), dtype=torch.float32)
    for contour, (start, end) in enumerate(CONTOUR_RANGES):
        contour_distances = distances[:, start:end]
        mask = torch.isfinite(contour_distances) & (
            contour_distances <= radius_mm
        )
        weights = torch.where(
            mask,
            torch.exp(
                -contour_distances.square() / (2.0 * sigma_mm * sigma_mm)
            ),
            torch.zeros_like(contour_distances),
        )
        denominator = weights.sum(dim=1)
        valid = denominator > 0.0
        if not valid.any():
            raise ValueError("curve arc radius contains no candidate for a contour")
        target = (
            weights
            * fractions[:, start:end, None]
        ).sum(dim=1) / denominator.clamp_min(1e-12)
        # Confidence saturates at one in overlap regions and tapers smoothly at
        # the edge of the supervised tube around the annotated curve.
        confidence = denominator.clamp(max=1.0) * valid
        error = F.smooth_l1_loss(
            arc_coordinates[:, contour].float(),
            target,
            reduction="none",
            beta=0.05,
        )
        weighted_error = weighted_error + (error * confidence).sum()
        total_weight = total_weight + confidence.sum()
    return weighted_error / total_weight.clamp_min(1.0)


def proposal_landmark_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale_mm,
    dense_surface: Optional[torch.Tensor] = None,
    anchor_weight: float = 0.0,
    spacing_weight: float = 0.0,
    surface_weight: float = 0.0,
    heatmap_logits: Optional[torch.Tensor] = None,
    heatmap_surface_points: Optional[torch.Tensor] = None,
    heatmap_weight: float = 0.0,
    heatmap_sigma_mm: float = 2.0,
    heatmap_geodesic_distances_mm: Optional[torch.Tensor] = None,
    vote_offsets: Optional[torch.Tensor] = None,
    vote_weight: float = 0.0,
    vote_radius_mm: float = 6.0,
    curve_logits: Optional[torch.Tensor] = None,
    curve_arc_coordinates: Optional[torch.Tensor] = None,
    curve_landmark_fractions: Optional[torch.Tensor] = None,
    curve_weight: float = 0.0,
    curve_arc_weight: float = 0.0,
    curve_sigma_mm: float = 3.0,
    curve_arc_radius_mm: float = 4.0,
) -> Mapping[str, torch.Tensor]:
    base = mean_distance_mm(prediction, target, scale_mm)
    anchor = anchor_distance_mm(prediction, target, scale_mm) if anchor_weight else base.new_zeros(())
    spacing = spacing_uniformity_mm(prediction, scale_mm) if spacing_weight else base.new_zeros(())
    if surface_weight:
        if dense_surface is None:
            raise ValueError("surface_weight requires dense_surface")
        surface = predicted_to_surface_mm(prediction, dense_surface, scale_mm)
    else:
        surface = base.new_zeros(())
    if heatmap_weight:
        if heatmap_logits is None or heatmap_surface_points is None:
            raise ValueError(
                "heatmap_weight requires heatmap logits and sampled surface points"
            )
        if heatmap_geodesic_distances_mm is None:
            heatmap = surface_heatmap_kl(
                heatmap_logits,
                heatmap_surface_points,
                target,
                scale_mm,
                heatmap_sigma_mm,
            )
        else:
            heatmap = surface_geodesic_heatmap_kl(
                heatmap_logits,
                heatmap_geodesic_distances_mm,
                heatmap_sigma_mm,
            )
    else:
        heatmap = base.new_zeros(())
    if vote_weight:
        if (
            vote_offsets is None
            or heatmap_surface_points is None
            or heatmap_geodesic_distances_mm is None
        ):
            raise ValueError(
                "vote_weight requires vote offsets, surface points, and geodesic distances"
            )
        vote = surface_vote_offset_loss(
            vote_offsets,
            heatmap_surface_points,
            target,
            scale_mm,
            heatmap_geodesic_distances_mm,
            vote_radius_mm,
            heatmap_sigma_mm,
        )
    else:
        vote = base.new_zeros(())
    if curve_weight:
        if curve_logits is None or heatmap_geodesic_distances_mm is None:
            raise ValueError(
                "curve_weight requires curve logits and geodesic distances"
            )
        curve = surface_curve_field_kl(
            curve_logits,
            heatmap_geodesic_distances_mm,
            curve_sigma_mm,
        )
    else:
        curve = base.new_zeros(())
    if curve_arc_weight:
        if (
            curve_arc_coordinates is None
            or curve_landmark_fractions is None
            or heatmap_geodesic_distances_mm is None
        ):
            raise ValueError(
                "curve_arc_weight requires arc coordinates, landmark fractions, "
                "and geodesic distances"
            )
        curve_arc = surface_curve_arc_loss(
            curve_arc_coordinates,
            heatmap_geodesic_distances_mm,
            curve_landmark_fractions,
            curve_sigma_mm,
            curve_arc_radius_mm,
        )
    else:
        curve_arc = base.new_zeros(())
    total = (
        base
        + anchor_weight * anchor
        + spacing_weight * spacing
        + surface_weight * surface
        + heatmap_weight * heatmap
        + vote_weight * vote
        + curve_weight * curve
        + curve_arc_weight * curve_arc
    )
    return {
        "total": total,
        "mean_distance": base,
        "anchor": anchor,
        "spacing": spacing,
        "surface": surface,
        "heatmap": heatmap,
        "vote": vote,
        "curve": curve,
        "curve_arc": curve_arc,
    }


def candidate_is_promoted(
    baseline_by_fold: Sequence[float],
    candidate_by_fold: Sequence[float],
    baseline_runtime_ms: float,
    candidate_runtime_ms: float,
) -> bool:
    if len(baseline_by_fold) != 5 or len(candidate_by_fold) != 5:
        raise ValueError("promotion requires exactly five outer-fold results")
    improved_folds = sum(c < b for b, c in zip(baseline_by_fold, candidate_by_fold))
    baseline_mean = sum(baseline_by_fold) / 5.0
    candidate_mean = sum(candidate_by_fold) / 5.0
    if candidate_mean < baseline_mean and improved_folds >= 3:
        return True
    if round(candidate_mean, 3) == round(baseline_mean, 3) and improved_folds >= 3:
        return candidate_runtime_ms < baseline_runtime_ms
    return False
