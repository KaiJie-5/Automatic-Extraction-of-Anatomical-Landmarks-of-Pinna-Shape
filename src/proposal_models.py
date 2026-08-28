"""Proposal-aligned locator, contour heads, and local landmark refinement."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .pointnet2_model import PointNet2FeatureEncoder, PointNet2LandmarkRegressor
from .pointnet2_utils import index_points
from .pointnext_model import PointNeXtEncoder


CONTOUR_LENGTHS = (25, 30, 20, 10)


def _make_encoder(backbone: str, config: Mapping[str, object]):
    name = backbone.lower()
    if name == "pointnet2":
        return PointNet2FeatureEncoder(**dict(config))
    if name == "pointnext":
        return PointNeXtEncoder(**dict(config))
    if name == "pointtransformerv3":
        from .pointtransformerv3_model import PointTransformerV3Encoder

        return PointTransformerV3Encoder(**dict(config))
    raise ValueError(f"unsupported point backbone: {backbone}")


class EarCenterLocator(nn.Module):
    def __init__(
        self,
        backbone: str = "pointnet2",
        encoder_config: Mapping[str, object] | None = None,
        head_channels: Sequence[int] = (256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = _make_encoder(backbone, encoder_config or {})
        self.head = PointNet2LandmarkRegressor._build_regression_head(
            self.encoder.feature_dim, head_channels, 3, dropout
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(points))


class LocalLandmarkRefiner(nn.Module):
    def __init__(
        self,
        k: int,
        offset_cap_normalized: float,
        embedding_dim: int = 8,
        contour_embedding_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if k not in {32, 64}:
            raise ValueError("local refinement k must be 32 or 64")
        self.k = int(k)
        self.offset_cap_normalized = float(offset_cap_normalized)
        self.identity = nn.Embedding(85, embedding_dim)
        self.contour_identity = nn.Embedding(4, contour_embedding_dim)
        contour_ids = torch.repeat_interleave(
            torch.arange(4), torch.as_tensor(CONTOUR_LENGTHS)
        )
        self.register_buffer("contour_ids", contour_ids, persistent=False)
        self.local_mlp = nn.Sequential(
            nn.Linear(7 + embedding_dim + contour_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Linear(hidden_dim, 3)

    def forward(self, coarse: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.shape[-1] != 6:
            point_cloud = point_cloud.transpose(1, 2)
        xyz = point_cloud[..., :3].float()
        normals = point_cloud[..., 3:6].float()
        coarse_float = coarse.float()
        indices = torch.cdist(coarse_float, xyz).topk(
            min(self.k, xyz.shape[1]), dim=-1, largest=False
        ).indices
        neighbor_xyz = index_points(xyz, indices)
        neighbor_normals = index_points(normals, indices)
        relative = neighbor_xyz - coarse_float.unsqueeze(2)
        distance = torch.linalg.norm(relative, dim=-1, keepdim=True)
        identities = self.identity(torch.arange(85, device=coarse.device))
        identities = identities.view(1, 85, 1, -1).expand(
            coarse.shape[0], -1, indices.shape[-1], -1
        )
        contours = self.contour_identity(self.contour_ids)
        contours = contours.view(1, 85, 1, -1).expand(
            coarse.shape[0], -1, indices.shape[-1], -1
        )
        features = torch.cat(
            [relative, neighbor_normals, distance, identities, contours], dim=-1
        )
        pooled = self.local_mlp(features).amax(dim=2)
        offset = torch.tanh(self.offset_head(pooled)) * self.offset_cap_normalized
        return coarse_float + offset


class ProposalLandmarkRegressor(nn.Module):
    def __init__(
        self,
        backbone: str = "pointnet2",
        encoder_config: Mapping[str, object] | None = None,
        four_heads: bool = True,
        head_channels: Sequence[int] = (512, 256),
        dropout: float = 0.0,
        refinement_k: int = 0,
        refinement_cap_normalized: float = 0.0,
    ):
        super().__init__()
        self.encoder = _make_encoder(backbone, encoder_config or {})
        self.four_heads = bool(four_heads)
        if self.four_heads:
            self.heads = nn.ModuleList(
                [
                    PointNet2LandmarkRegressor._build_regression_head(
                        self.encoder.feature_dim, head_channels, length * 3, dropout
                    )
                    for length in CONTOUR_LENGTHS
                ]
            )
            self.head = None
        else:
            self.head = PointNet2LandmarkRegressor._build_regression_head(
                self.encoder.feature_dim, head_channels, 85 * 3, dropout
            )
            self.heads = nn.ModuleList()
        self.refiner = (
            LocalLandmarkRefiner(refinement_k, refinement_cap_normalized)
            if refinement_k
            else None
        )

    def _coarse_prediction(self, points: torch.Tensor) -> torch.Tensor:
        global_features = self.encoder(points)
        if self.four_heads:
            chunks = [
                head(global_features).view(points.shape[0], length, 3)
                for head, length in zip(self.heads, CONTOUR_LENGTHS)
            ]
            coarse = torch.cat(chunks, dim=1)
        else:
            coarse = self.head(global_features).view(points.shape[0], 85, 3)
        return coarse

    def forward_with_details(self, points: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return coarse and final outputs without changing the training forward API."""
        coarse = self._coarse_prediction(points)
        final = self.refiner(coarse, points) if self.refiner is not None else coarse
        return {"coarse": coarse, "final": final}

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(points)["final"]


def build_locator(config: Mapping[str, object]) -> EarCenterLocator:
    return EarCenterLocator(**dict(config))


def build_landmark_model(config: Mapping[str, object]) -> ProposalLandmarkRegressor:
    return ProposalLandmarkRegressor(**dict(config))


def build_fold_landmark_model(config: Mapping[str, object]) -> nn.Module:
    """Build any landmark backbone accepted by the proposal fold trainer."""
    values = dict(config)
    if values.get("backbone") == "meshnet":
        from .meshnet import MeshNetLandmarkRegressor

        values.pop("backbone")
        values.pop("target_faces")
        return MeshNetLandmarkRegressor(**values)
    return build_landmark_model(values)
