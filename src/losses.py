"""Proposal-aligned landmark objectives and experiment promotion."""

from __future__ import annotations

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


def proposal_landmark_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale_mm,
    dense_surface: Optional[torch.Tensor] = None,
    anchor_weight: float = 0.0,
    spacing_weight: float = 0.0,
    surface_weight: float = 0.0,
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
    total = base + anchor_weight * anchor + spacing_weight * spacing + surface_weight * surface
    return {"total": total, "mean_distance": base, "anchor": anchor, "spacing": spacing, "surface": surface}


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
