"""Portable PyTorch PointNeXt-S-style encoder.

This implementation deliberately reuses the repository's PointNet++ sampling and
grouping primitives. It avoids compiled OpenPoints/CUDA extensions so checkpoints
remain usable in the challenge evaluator.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .pointnet2_utils import farthest_point_sample, index_points, square_distance


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


class PointNeXtEncoder(nn.Module):
    """C32/B0-style five-stage global encoder for crop-local point features."""

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
    ):
        super().__init__()
        if tuple(blocks) != (1, 1, 1, 1, 1):
            raise ValueError("portable PointNeXt currently implements the C32/B0 block layout")
        if len(strides) != 5:
            raise ValueError("PointNeXt requires five stride entries")
        self.input_channels = int(input_channels)
        self.stem = _MLP(input_channels, width, expansion=1)
        channels = [width, width * 2, width * 4, width * 8, width * 16]
        stages = []
        input_dim = width
        for stage_index, (output_dim, stride) in enumerate(zip(channels, strides)):
            stages.append(
                PointNeXtSetAbstraction(
                    input_dim,
                    output_dim,
                    stride=stride,
                    radius=radius * (radius_scaling ** max(stage_index - 1, 0)),
                    nsample=nsample,
                    expansion=expansion,
                )
            )
            input_dim = output_dim
        self.stages = nn.ModuleList(stages)
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
        for stage in self.stages:
            xyz, features = stage(xyz, features)
        return features.amax(dim=1)


def default_pointnext_config() -> dict:
    return {
        "input_channels": 6,
        "width": 32,
        "strides": [1, 4, 4, 4, 4],
        "blocks": [1, 1, 1, 1, 1],
        "radius": 0.1,
        "radius_scaling": 2.0,
        "nsample": 32,
        "expansion": 4,
    }
