from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def default_model_config() -> dict:
    return {
        "num_landmarks": 85,
        "num_kernel": 64,
        "sigma": 0.2,
        "aggregation_method": "Concat",
        "mask_ratio": 0.95,
        "dropout": 0.5,
        "head_channels": [512, 256],
    }


class FaceRotateConvolution(nn.Module):
    """Rotate over the three corner-vector pairs and fuse (paper Sec. 3.2)."""

    def __init__(self):
        super().__init__()
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        fea = (
            self.rotate_mlp(corners[:, :6])
            + self.rotate_mlp(corners[:, 3:9])
            + self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))
        ) / 3
        return self.fusion_mlp(fea)


class FaceKernelCorrelation(nn.Module):
    """Correlate face + neighbor normals with learnable kernels (paper Sec. 3.2)."""

    def __init__(self, num_kernel: int = 64, sigma: float = 0.2):
        super().__init__()
        self.num_kernel = num_kernel
        self.sigma = sigma
        self.weight_alpha = Parameter(torch.rand(1, num_kernel, 4) * np.pi)
        self.weight_beta = Parameter(torch.rand(1, num_kernel, 4) * 2 * np.pi)
        self.bn = nn.BatchNorm1d(num_kernel)
        self.relu = nn.ReLU()

    def forward(self, normals: torch.Tensor, neighbor_index: torch.Tensor) -> torch.Tensor:
        b, _, n = normals.size()

        center = normals.unsqueeze(2).expand(-1, -1, self.num_kernel, -1).unsqueeze(4)
        neighbor = torch.gather(
            normals.unsqueeze(3).expand(-1, -1, -1, 3),
            2,
            neighbor_index.unsqueeze(1).expand(-1, 3, -1, -1),
        )
        neighbor = neighbor.unsqueeze(2).expand(-1, -1, self.num_kernel, -1, -1)

        fea = torch.cat([center, neighbor], 4)
        fea = fea.unsqueeze(5).expand(-1, -1, -1, -1, -1, 4)
        weight = torch.cat(
            [
                torch.sin(self.weight_alpha) * torch.cos(self.weight_beta),
                torch.sin(self.weight_alpha) * torch.sin(self.weight_beta),
                torch.cos(self.weight_alpha),
            ],
            0,
        )
        weight = weight.unsqueeze(0).expand(b, -1, -1, -1)
        weight = weight.unsqueeze(3).expand(-1, -1, -1, n, -1)
        weight = weight.unsqueeze(4).expand(-1, -1, -1, -1, 4, -1)

        dist = torch.sum((fea - weight) ** 2, 1)
        fea = torch.sum(torch.sum(np.e ** (dist / (-2 * self.sigma**2)), 4), 3) / 16

        return self.relu(self.bn(fea))


class SpatialDescriptor(nn.Module):
    """Shared MLP over face center positions (paper Sec. 3.2)."""

    def __init__(self):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        return self.spatial_mlp(centers)


class StructuralDescriptor(nn.Module):
    """Face Rotate Convolution + Face Kernel Correlation + normals."""

    def __init__(self, num_kernel: int = 64, sigma: float = 0.2):
        super().__init__()
        self.FRC = FaceRotateConvolution()
        self.FKC = FaceKernelCorrelation(num_kernel, sigma)
        self.structural_mlp = nn.Sequential(
            nn.Conv1d(64 + 3 + num_kernel, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
            nn.Conv1d(131, 131, 1),
            nn.BatchNorm1d(131),
            nn.ReLU(),
        )

    def forward(
        self, corners: torch.Tensor, normals: torch.Tensor, neighbor_index: torch.Tensor
    ) -> torch.Tensor:
        structural_fea1 = self.FRC(corners)
        structural_fea2 = self.FKC(normals, neighbor_index)
        return self.structural_mlp(torch.cat([structural_fea1, structural_fea2, normals], 1))


class MeshConvolution(nn.Module):
    """Combination (spatial+structural) and Aggregation (neighbors) block."""

    def __init__(
        self,
        spatial_in_channel: int,
        structural_in_channel: int,
        spatial_out_channel: int,
        structural_out_channel: int,
        aggregation_method: str = "Concat",
    ):
        super().__init__()
        self.spatial_in_channel = spatial_in_channel
        self.structural_in_channel = structural_in_channel
        self.spatial_out_channel = spatial_out_channel
        self.structural_out_channel = structural_out_channel

        if aggregation_method not in ("Concat", "Max", "Average"):
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        self.aggregation_method = aggregation_method

        self.combination_mlp = nn.Sequential(
            nn.Conv1d(
                self.spatial_in_channel + self.structural_in_channel,
                self.spatial_out_channel,
                1,
            ),
            nn.BatchNorm1d(self.spatial_out_channel),
            nn.ReLU(),
        )

        if self.aggregation_method == "Concat":
            self.concat_mlp = nn.Sequential(
                nn.Conv2d(self.structural_in_channel * 2, self.structural_in_channel, 1),
                nn.BatchNorm2d(self.structural_in_channel),
                nn.ReLU(),
            )

        self.aggregation_mlp = nn.Sequential(
            nn.Conv1d(self.structural_in_channel, self.structural_out_channel, 1),
            nn.BatchNorm1d(self.structural_out_channel),
            nn.ReLU(),
        )

    def forward(self, spatial_fea, structural_fea, neighbor_index):
        b, _, n = spatial_fea.size()

        # Combination
        spatial_fea = self.combination_mlp(torch.cat([spatial_fea, structural_fea], 1))

        # Aggregation
        if self.aggregation_method == "Concat":
            structural_fea = torch.cat(
                [
                    structural_fea.unsqueeze(3).expand(-1, -1, -1, 3),
                    torch.gather(
                        structural_fea.unsqueeze(3).expand(-1, -1, -1, 3),
                        2,
                        neighbor_index.unsqueeze(1).expand(
                            -1, self.structural_in_channel, -1, -1
                        ),
                    ),
                ],
                1,
            )
            structural_fea = self.concat_mlp(structural_fea)
            structural_fea = torch.max(structural_fea, 3)[0]
        elif self.aggregation_method == "Max":
            structural_fea = torch.cat(
                [
                    structural_fea.unsqueeze(3),
                    torch.gather(
                        structural_fea.unsqueeze(3).expand(-1, -1, -1, 3),
                        2,
                        neighbor_index.unsqueeze(1).expand(
                            -1, self.structural_in_channel, -1, -1
                        ),
                    ),
                ],
                3,
            )
            structural_fea = torch.max(structural_fea, 3)[0]
        else:  # Average
            structural_fea = torch.cat(
                [
                    structural_fea.unsqueeze(3),
                    torch.gather(
                        structural_fea.unsqueeze(3).expand(-1, -1, -1, 3),
                        2,
                        neighbor_index.unsqueeze(1).expand(
                            -1, self.structural_in_channel, -1, -1
                        ),
                    ),
                ],
                3,
            )
            structural_fea = torch.sum(structural_fea, dim=3) / 4

        structural_fea = self.aggregation_mlp(structural_fea)
        return spatial_fea, structural_fea


class MeshNetLandmarkRegressor(nn.Module):
    """MeshNet backbone with the classifier replaced by a landmark regression head.

    Input (per batch):
        centers:        (B, 3, F)  face center positions
        corners:        (B, 9, F)  corner vectors relative to the face center
        normals:        (B, 3, F)  unit face normals
        neighbor_index: (B, F, 3)  indices of the 3 edge-adjacent faces

    Output:
        (B, num_landmarks, 3) landmark coordinates in normalized space.
    """

    def __init__(
        self,
        num_landmarks: int = 85,
        num_kernel: int = 64,
        sigma: float = 0.2,
        aggregation_method: str = "Concat",
        mask_ratio: float = 0.95,
        dropout: float = 0.5,
        head_channels: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.num_landmarks = int(num_landmarks)
        self.mask_ratio = float(mask_ratio)
        head_channels = list(head_channels) if head_channels else [512, 256]

        self.spatial_descriptor = SpatialDescriptor()
        self.structural_descriptor = StructuralDescriptor(num_kernel, sigma)
        self.mesh_conv1 = MeshConvolution(64, 131, 256, 256, aggregation_method)
        self.mesh_conv2 = MeshConvolution(256, 256, 512, 512, aggregation_method)
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.concat_mlp = nn.Sequential(
            nn.Conv1d(1792, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        # Regression head replacing the original MeshNet classifier.
        layers: List[nn.Module] = []
        in_channels = 1024
        for out_channels in head_channels:
            layers.append(nn.Linear(in_channels, out_channels))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_channels = out_channels
        layers.append(nn.Linear(in_channels, self.num_landmarks * 3))
        self.regression_head = nn.Sequential(*layers)

    def forward(
        self,
        centers: torch.Tensor,
        corners: torch.Tensor,
        normals: torch.Tensor,
        neighbor_index: torch.Tensor,
    ) -> torch.Tensor:
        spatial_fea0 = self.spatial_descriptor(centers)
        structural_fea0 = self.structural_descriptor(corners, normals, neighbor_index)

        spatial_fea1, structural_fea1 = self.mesh_conv1(
            spatial_fea0, structural_fea0, neighbor_index
        )
        spatial_fea2, structural_fea2 = self.mesh_conv2(
            spatial_fea1, structural_fea1, neighbor_index
        )
        spatial_fea3 = self.fusion_mlp(torch.cat([spatial_fea2, structural_fea2], 1))

        fea = self.concat_mlp(torch.cat([spatial_fea1, spatial_fea2, spatial_fea3], 1))
        if self.training and self.mask_ratio > 0:
            keep = int(fea.size(2) * (1 - self.mask_ratio))
            keep = max(keep, 1)
            fea = fea[:, :, torch.randperm(fea.size(2), device=fea.device)[:keep]]
        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)

        pred = self.regression_head(fea)
        return pred.view(-1, self.num_landmarks, 3)