"""PointNet++ set abstraction utilities.

Adapted from yanx27/Pointnet_Pointnet2_pytorch:
https://github.com/yanx27/Pointnet_Pointnet2_pytorch

The upstream project is MIT licensed. See LICENSE text in that repository.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distance.

    Args:
        src: Source points with shape (B, N, C).
        dst: Target points with shape (B, M, C).

    Returns:
        Pairwise squared distances with shape (B, N, M).
    """
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, dim=-1).unsqueeze(-1)
    dist += torch.sum(dst**2, dim=-1).unsqueeze(1)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by batched indices."""
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    batch_indices = batch_indices.view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling on xyz coordinates.

    Args:
        xyz: Point coordinates with shape (B, N, 3).
        npoint: Number of centroids to sample.

    Returns:
        Sampled centroid indices with shape (B, npoint).
    """
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Find local neighborhoods within radius around sampled centroids."""
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, num_centroids, _ = new_xyz.shape

    group_idx = torch.arange(num_points, dtype=torch.long, device=device)
    group_idx = group_idx.view(1, 1, num_points).repeat(batch_size, num_centroids, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(batch_size, num_centroids, 1)
    group_first = group_first.repeat(1, 1, nsample)
    mask = group_idx == num_points
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample centroids and group their local neighborhoods."""
    batch_size, num_points, channels = xyz.shape
    effective_nsample = min(nsample, num_points)
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, effective_nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(batch_size, npoint, 1, channels)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor, points: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group the full point cloud into a single global neighborhood."""
    device = xyz.device
    batch_size, num_points, channels = xyz.shape
    new_xyz = torch.zeros(batch_size, 1, channels, device=device)
    grouped_xyz = xyz.view(batch_size, 1, num_points, channels)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(batch_size, 1, num_points, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """Single-scale PointNet++ set abstraction layer."""

    def __init__(
        self,
        npoint: Optional[int],
        radius: Optional[float],
        nsample: Optional[int],
        in_channel: int,
        mlp: List[int],
        group_all: bool,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run set abstraction.

        Args:
            xyz: Coordinates with shape (B, 3, N).
            points: Optional point features with shape (B, D, N).

        Returns:
            New coordinates (B, 3, S) and features (B, D_out, S).
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            if self.npoint is None or self.radius is None or self.nsample is None:
                raise ValueError("npoint, radius, and nsample are required unless group_all=True")
            effective_npoint = min(self.npoint, xyz.shape[1])
            new_xyz, new_points = sample_and_group(
                effective_npoint, self.radius, self.nsample, xyz, points
            )

        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """Multi-scale PointNet++ set abstraction layer."""

    def __init__(
        self,
        npoint: int,
        radius_list: List[float],
        nsample_list: List[int],
        in_channel: int,
        mlp_list: List[List[int]],
    ):
        super().__init__()
        if not (len(radius_list) == len(nsample_list) == len(mlp_list)):
            raise ValueError("radius_list, nsample_list, and mlp_list must have equal length")

        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for mlp in mlp_list:
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(
        self, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run multi-scale set abstraction."""
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        batch_size, num_points, channels = xyz.shape
        effective_npoint = min(self.npoint, num_points)
        new_xyz = index_points(xyz, farthest_point_sample(xyz, effective_npoint))
        new_points_list = []

        for scale_idx, radius in enumerate(self.radius_list):
            nsample = min(self.nsample_list[scale_idx], num_points)
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz = grouped_xyz - new_xyz.view(batch_size, effective_npoint, 1, channels)

            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)
            for conv, bn in zip(self.conv_blocks[scale_idx], self.bn_blocks[scale_idx]):
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, dim=2)[0]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, torch.cat(new_points_list, dim=1)
