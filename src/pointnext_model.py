"""Portable PyTorch PointNeXt S/B/L/XL-style encoder.

This implementation deliberately reuses the repository's PointNet++ sampling and
grouping primitives. It avoids compiled OpenPoints/CUDA extensions so checkpoints
remain usable in the challenge evaluator.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .pointnet2_utils import farthest_point_sample, index_points, square_distance


POINTNEXT_VARIANT_BLOCKS = {
    "s": (1, 1, 1, 1, 1),
    "b": (1, 2, 3, 2, 2),
    "l": (1, 3, 5, 3, 3),
    "xl": (1, 4, 7, 4, 4),
}

POINTNEXT_VARIANT_WIDTHS = {
    "s": 32,
    "b": 32,
    "l": 32,
    "xl": 64,
}


def _chunked_ball_query(radius, nsample, xyz, centers, chunk_size=512):
    """Memory-bounded ball query for 16k-point portable operation."""
    groups = []
    count = min(int(nsample), xyz.shape[1])
    for start in range(0, centers.shape[1], chunk_size):
        chunk = centers[:, start : start + chunk_size]
        distances = square_distance(chunk, xyz)
        masked = distances.masked_fill(distances > float(radius) ** 2, float("inf"))
        values, indices = masked.topk(count, dim=-1, largest=False, sorted=False)
        nearest = distances.argmin(dim=-1, keepdim=True).expand_as(indices)
        indices = torch.where(torch.isfinite(values), indices, nearest)
        groups.append(indices)
    return torch.cat(groups, dim=1)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, expansion: int = 4):
        super().__init__()
        hidden = max(output_dim, input_dim * expansion)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class PointNeXtSetAbstraction(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        radius: float,
        nsample: int = 32,
        expansion: int = 4,
    ):
        super().__init__()
        self.stride = int(stride)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.local = _MLP(input_dim + 3, output_dim, expansion=expansion)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        if self.stride == 1:
            relative = torch.zeros_like(xyz)
            aggregated = self.local(torch.cat([relative, features], dim=-1))
            return xyz, aggregated + self.residual(features)
        npoint = max(1, xyz.shape[1] // self.stride)
        center_indices = farthest_point_sample(xyz, npoint)
        centers = index_points(xyz, center_indices)
        center_features = index_points(features, center_indices)
        group_indices = _chunked_ball_query(
            self.radius, min(self.nsample, xyz.shape[1]), xyz, centers
        )
        grouped_xyz = index_points(xyz, group_indices)
        grouped_features = index_points(features, group_indices)
        relative = (grouped_xyz - centers.unsqueeze(2)) / max(self.radius, 1e-8)
        aggregated = self.local(torch.cat([relative, grouped_features], dim=-1)).amax(dim=2)
        return centers, aggregated + self.residual(center_features)


class PointNeXtInvertedResidualBlock(nn.Module):
    """Portable same-resolution PointNeXt inverted-residual local block."""

    def __init__(
        self,
        channels: int,
        radius: float,
        nsample: int = 32,
        expansion: int = 4,
    ):
        super().__init__()
        channels = int(channels)
        hidden = channels * int(expansion)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.local = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.LayerNorm(channels),
        )
        self.activation = nn.GELU()

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        group_indices = _chunked_ball_query(
            self.radius, min(self.nsample, xyz.shape[1]), xyz, xyz
        )
        grouped_xyz = index_points(xyz, group_indices)
        grouped_features = index_points(features, group_indices)
        relative = (grouped_xyz - xyz.unsqueeze(2)) / max(self.radius, 1e-8)
        local = self.local(torch.cat([relative, grouped_features], dim=-1)).amax(dim=2)
        return self.activation(features + self.pointwise(local))


class PointNeXtEncoder(nn.Module):
    """Variant- and width-tunable five-stage crop-local point encoder."""

    def __init__(
        self,
        input_channels: int = 6,
        width: int = 32,
        strides: Sequence[int] = (1, 4, 4, 4, 4),
        blocks: Sequence[int] = (1, 1, 1, 1, 1),
        radius: float = 0.1,
        radius_scaling: float = 2.0,
        nsample: int = 32,
        expansion: int = 4,
        variant: str | None = None,
    ):
        super().__init__()
        width = int(width)
        if width <= 0:
            raise ValueError("PointNeXt width must be a positive integer")
        blocks = tuple(int(value) for value in blocks)
        if len(blocks) != 5 or any(value < 1 for value in blocks):
            raise ValueError("PointNeXt blocks must contain five positive integers")
        if variant is not None:
            variant = str(variant).lower()
            if variant not in POINTNEXT_VARIANT_BLOCKS:
                raise ValueError(f"unsupported PointNeXt variant: {variant}")
            if blocks != POINTNEXT_VARIANT_BLOCKS[variant]:
                raise ValueError(
                    f"PointNeXt-{variant.upper()} requires blocks "
                    f"{list(POINTNEXT_VARIANT_BLOCKS[variant])}"
                )
        if len(strides) != 5:
            raise ValueError("PointNeXt requires five stride entries")
        self.input_channels = int(input_channels)
        self.variant = variant
        self.blocks = blocks
        self.stem = _MLP(input_channels, width, expansion=1)
        channels = [width, width * 2, width * 4, width * 8, width * 16]
        stages = []
        residual_stages = []
        input_dim = width
        for stage_index, (output_dim, stride, block_count) in enumerate(
            zip(channels, strides, blocks)
        ):
            stage_radius = radius * (radius_scaling ** max(stage_index - 1, 0))
            stages.append(
                PointNeXtSetAbstraction(
                    input_dim,
                    output_dim,
                    stride=stride,
                    radius=stage_radius,
                    nsample=nsample,
                    expansion=expansion,
                )
            )
            residual_radius = (
                stage_radius * radius_scaling if int(stride) != 1 else stage_radius
            )
            residual_stages.append(
                nn.ModuleList(
                    [
                        PointNeXtInvertedResidualBlock(
                            output_dim,
                            radius=residual_radius,
                            nsample=nsample,
                            expansion=expansion,
                        )
                        for _ in range(block_count - 1)
                    ]
                )
            )
            input_dim = output_dim
        self.stages = nn.ModuleList(stages)
        self.residual_stages = nn.ModuleList(residual_stages)
        self.feature_dim = channels[-1]

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3:
            raise ValueError("point_cloud must have shape (B, N, C) or (B, C, N)")
        if point_cloud.shape[-1] != self.input_channels:
            if point_cloud.shape[1] == self.input_channels:
                point_cloud = point_cloud.transpose(1, 2)
            else:
                raise ValueError(f"expected {self.input_channels} input channels")
        xyz = point_cloud[..., :3]
        features = self.stem(point_cloud)
        for stage, residual_blocks in zip(self.stages, self.residual_stages):
            xyz, features = stage(xyz, features)
            for block in residual_blocks:
                features = block(xyz, features)
        return features.amax(dim=1)


def default_pointnext_config(width: int | None = None, variant: str = "s") -> dict:
    variant = str(variant).lower()
    if variant not in POINTNEXT_VARIANT_BLOCKS:
        raise ValueError(f"unsupported PointNeXt variant: {variant}")
    width = POINTNEXT_VARIANT_WIDTHS[variant] if width is None else int(width)
    if width <= 0:
        raise ValueError("PointNeXt width must be a positive integer")
    return {
        "input_channels": 6,
        "width": width,
        "strides": [1, 4, 4, 4, 4],
        "blocks": list(POINTNEXT_VARIANT_BLOCKS[variant]),
        "radius": 0.1,
        "radius_scaling": 2.0,
        "nsample": 32,
        "expansion": 4,
        "variant": variant,
    }
