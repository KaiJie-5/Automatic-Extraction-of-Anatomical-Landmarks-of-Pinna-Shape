"""Proposal-aligned landmark objectives and experiment promotion."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import torch


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
    return (
        target_probabilities
        * (target_log_probabilities - predicted_log_probabilities)
    ).sum(dim=-1).mean()


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
        heatmap = surface_heatmap_kl(
            heatmap_logits,
            heatmap_surface_points,
            target,
            scale_mm,
            heatmap_sigma_mm,
        )
    else:
        heatmap = base.new_zeros(())
    total = (
        base
        + anchor_weight * anchor
        + spacing_weight * spacing
        + surface_weight * surface
        + heatmap_weight * heatmap
    )
    return {
        "total": total,
        "mean_distance": base,
        "anchor": anchor,
        "spacing": spacing,
        "surface": surface,
        "heatmap": heatmap,
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
