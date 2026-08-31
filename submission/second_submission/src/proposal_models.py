"""Proposal-aligned locator, contour heads, and local landmark refinement."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .pointnet2_model import PointNet2FeatureEncoder, PointNet2LandmarkRegressor
from .pointnet2_utils import index_points, square_distance
from .pointnext_model import PointNeXtEncoder


CONTOUR_LENGTHS = (25, 30, 20, 10)
REFINEMENT_ANCHOR_MODES = ("raw", "nearest-surface-sample")


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
        anchor_mode: str = "raw",
    ):
        super().__init__()
        if k not in {32, 64}:
            raise ValueError("local refinement k must be 32 or 64")
        anchor_mode = str(anchor_mode).lower()
        if anchor_mode not in REFINEMENT_ANCHOR_MODES:
            raise ValueError(
                "local refinement anchor_mode must be one of "
                f"{REFINEMENT_ANCHOR_MODES}"
            )
        self.k = int(k)
        self.offset_cap_normalized = float(offset_cap_normalized)
        self.anchor_mode = anchor_mode
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

    def _query_centers(
        self, coarse: torch.Tensor, xyz: torch.Tensor
    ) -> torch.Tensor:
        if self.anchor_mode == "raw":
            return coarse
        nearest_indices = torch.cdist(coarse, xyz).argmin(dim=-1)
        return index_points(xyz, nearest_indices)

    def forward(self, coarse: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.shape[-1] != 6:
            point_cloud = point_cloud.transpose(1, 2)
        xyz = point_cloud[..., :3].float()
        normals = point_cloud[..., 3:6].float()
        coarse_float = coarse.float()
        query_centers = self._query_centers(coarse_float, xyz)
        indices = torch.cdist(query_centers, xyz).topk(
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


def _three_neighbour_interpolate(
    fine_xyz: torch.Tensor,
    coarse_xyz: torch.Tensor,
    coarse_features: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Deterministic, memory-bounded inverse-distance feature interpolation."""
    if fine_xyz.ndim != 3 or coarse_xyz.ndim != 3 or coarse_features.ndim != 3:
        raise ValueError("feature interpolation inputs must be batched tensors")
    if coarse_xyz.shape[:2] != coarse_features.shape[:2]:
        raise ValueError("coarse XYZ and features must have matching point axes")
    neighbour_count = min(3, int(coarse_xyz.shape[1]))
    if neighbour_count <= 0:
        raise ValueError("cannot interpolate from an empty point set")
    outputs = []
    for start in range(0, fine_xyz.shape[1], int(chunk_size)):
        query = fine_xyz[:, start : start + int(chunk_size)]
        squared = square_distance(query, coarse_xyz)
        values, indices = squared.topk(
            neighbour_count, dim=-1, largest=False, sorted=True
        )
        # Use Euclidean inverse distance.  The exact-match clamp also makes
        # interpolation well defined when an FPS centre survives at both levels.
        inverse = torch.rsqrt(values.clamp_min(1e-12))
        weights = inverse / inverse.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        neighbours = index_points(coarse_features, indices)
        outputs.append((neighbours * weights.unsqueeze(-1)).sum(dim=2))
    return torch.cat(outputs, dim=1)


class PointNeXtFeaturePropagation(nn.Module):
    """Interpolate one PointNeXt level and fuse it with its encoder skip."""

    def __init__(self, skip_channels: int, coarse_channels: int, output_channels: int):
        super().__init__()
        hidden = max(int(output_channels), int((skip_channels + coarse_channels) // 2))
        self.net = nn.Sequential(
            nn.Linear(int(skip_channels + coarse_channels), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, int(output_channels)),
            nn.LayerNorm(int(output_channels)),
            nn.GELU(),
        )

    def forward(
        self,
        fine_xyz: torch.Tensor,
        fine_features: torch.Tensor,
        coarse_xyz: torch.Tensor,
        coarse_features: torch.Tensor,
    ) -> torch.Tensor:
        interpolated = _three_neighbour_interpolate(
            fine_xyz, coarse_xyz, coarse_features
        )
        return self.net(torch.cat([fine_features, interpolated], dim=-1))


class PointNeXtSurfaceHeatmapRegressor(nn.Module):
    """Surface-candidate heatmaps with PointNeXt feature propagation.

    All 85 outputs are weighted combinations of sampled crop-surface points
    before the optional legacy local refiner.  Four independent query branches
    preserve the official contour ordering when ``four_heads`` is enabled.
    """

    def __init__(
        self,
        encoder_config: Mapping[str, object] | None = None,
        four_heads: bool = True,
        heatmap_feature_dim: int = 128,
        heatmap_topk: int = 64,
        heatmap_coordinate_temperature: float = 1.0,
        refinement_k: int = 0,
        refinement_cap_normalized: float = 0.0,
        refinement_anchor: str = "raw",
    ):
        super().__init__()
        heatmap_feature_dim = int(heatmap_feature_dim)
        heatmap_topk = int(heatmap_topk)
        heatmap_coordinate_temperature = float(heatmap_coordinate_temperature)
        if heatmap_feature_dim <= 0 or heatmap_topk <= 0:
            raise ValueError("surface heatmap dimensions and top-k must be positive")
        if not math.isfinite(heatmap_coordinate_temperature) or heatmap_coordinate_temperature <= 0:
            raise ValueError("surface heatmap coordinate temperature must be positive")
        refinement_anchor = str(refinement_anchor).lower()
        if refinement_anchor not in REFINEMENT_ANCHOR_MODES:
            raise ValueError(
                "refinement_anchor must be one of "
                f"{REFINEMENT_ANCHOR_MODES}"
            )
        if not refinement_k and refinement_anchor != "raw":
            raise ValueError(
                "non-raw refinement_anchor requires refinement_k to be enabled"
            )
        self.encoder = PointNeXtEncoder(**dict(encoder_config or {}))
        channels = tuple(int(value) for value in self.encoder.stage_channels)
        if len(channels) != 5:
            raise ValueError("surface heatmap decoder requires five PointNeXt levels")
        self.four_heads = bool(four_heads)
        self.heatmap_topk = heatmap_topk
        self.heatmap_coordinate_temperature = heatmap_coordinate_temperature

        # Decode the 64-point semantic level back to all original 16,384
        # sampled surface points using the exact encoder FPS hierarchy.
        self.propagation_3 = PointNeXtFeaturePropagation(
            channels[3], channels[4], channels[3]
        )
        self.propagation_2 = PointNeXtFeaturePropagation(
            channels[2], channels[3], channels[2]
        )
        self.propagation_1 = PointNeXtFeaturePropagation(
            channels[1], channels[2], channels[1]
        )
        self.propagation_0 = PointNeXtFeaturePropagation(
            channels[0], channels[1], heatmap_feature_dim
        )
        self.point_projection = nn.Sequential(
            nn.Linear(heatmap_feature_dim, heatmap_feature_dim),
            nn.LayerNorm(heatmap_feature_dim),
            nn.GELU(),
        )
        self.global_projection = nn.Linear(channels[4], heatmap_feature_dim)
        self.point_norm = nn.LayerNorm(heatmap_feature_dim)

        lengths = CONTOUR_LENGTHS if self.four_heads else (sum(CONTOUR_LENGTHS),)
        self.query_embeddings = nn.ParameterList(
            [nn.Parameter(torch.empty(length, heatmap_feature_dim)) for length in lengths]
        )
        self.query_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(heatmap_feature_dim, heatmap_feature_dim),
                    nn.LayerNorm(heatmap_feature_dim),
                )
                for _ in lengths
            ]
        )
        for queries in self.query_embeddings:
            nn.init.trunc_normal_(queries, std=0.02)
        self.logit_scale = heatmap_feature_dim ** -0.5
        self.refiner = (
            LocalLandmarkRefiner(
                refinement_k,
                refinement_cap_normalized,
                anchor_mode=refinement_anchor,
            )
            if refinement_k
            else None
        )

    def _decode_point_features(self, levels: Mapping[str, object]) -> torch.Tensor:
        xyz = levels["xyz"]
        features = levels["features"]
        decoded_3 = self.propagation_3(xyz[3], features[3], xyz[4], features[4])
        decoded_2 = self.propagation_2(xyz[2], features[2], xyz[3], decoded_3)
        decoded_1 = self.propagation_1(xyz[1], features[1], xyz[2], decoded_2)
        decoded_0 = self.propagation_0(xyz[0], features[0], xyz[1], decoded_1)
        global_context = self.global_projection(levels["global"]).unsqueeze(1)
        return self.point_norm(self.point_projection(decoded_0) + global_context)

    def _logits(self, point_features: torch.Tensor) -> torch.Tensor:
        chunks = []
        for queries, projection in zip(
            self.query_embeddings, self.query_projections
        ):
            query_features = projection(queries)
            chunks.append(
                torch.einsum("bnd,ld->bln", point_features, query_features)
                * self.logit_scale
            )
        return torch.cat(chunks, dim=1)

    def _surface_coordinates(
        self, logits: torch.Tensor, xyz: torch.Tensor
    ) -> torch.Tensor:
        count = min(self.heatmap_topk, int(xyz.shape[1]))
        values, indices = logits.topk(count, dim=-1, largest=True, sorted=True)
        candidates = index_points(xyz, indices)
        weights = torch.softmax(
            values.float() / self.heatmap_coordinate_temperature, dim=-1
        ).to(candidates.dtype)
        return (candidates * weights.unsqueeze(-1)).sum(dim=2).float()

    def forward_with_details(self, points: torch.Tensor) -> Mapping[str, torch.Tensor]:
        levels = self.encoder.forward_features(points)
        point_features = self._decode_point_features(levels)
        logits = self._logits(point_features)
        xyz = levels["xyz"][0]
        coarse = self._surface_coordinates(logits, xyz)
        final = self.refiner(coarse, points) if self.refiner is not None else coarse
        return {
            "coarse": coarse,
            "final": final,
            "heatmap_logits": logits,
            "surface_candidates": xyz,
        }

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(points)["final"]


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
        refinement_anchor: str = "raw",
    ):
        super().__init__()
        refinement_anchor = str(refinement_anchor).lower()
        if refinement_anchor not in REFINEMENT_ANCHOR_MODES:
            raise ValueError(
                "refinement_anchor must be one of "
                f"{REFINEMENT_ANCHOR_MODES}"
            )
        if not refinement_k and refinement_anchor != "raw":
            raise ValueError(
                "non-raw refinement_anchor requires refinement_k to be enabled"
            )
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
            LocalLandmarkRefiner(
                refinement_k,
                refinement_cap_normalized,
                anchor_mode=refinement_anchor,
            )
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


def build_landmark_model(config: Mapping[str, object]) -> nn.Module:
    values = dict(config)
    # Runtime precision is serialized beside the architecture so evaluation
    # can reproduce PTv3's required FP16 autocast, but it is not a constructor
    # argument of the landmark network itself.
    values.pop("amp_dtype", None)
    decoder = str(values.pop("decoder", "coordinate_regression"))
    if decoder == "surface_heatmap":
        backbone = str(values.pop("backbone", ""))
        if backbone != "pointnext":
            raise ValueError("surface heatmap decoder requires PointNeXt")
        return PointNeXtSurfaceHeatmapRegressor(**values)
    if decoder != "coordinate_regression":
        raise ValueError(f"unsupported landmark decoder: {decoder}")
    return ProposalLandmarkRegressor(**values)


def build_fold_landmark_model(config: Mapping[str, object]) -> nn.Module:
    """Build any landmark backbone accepted by the proposal fold trainer."""
    values = dict(config)
    values.pop("amp_dtype", None)
    if values.get("backbone") == "meshnet":
        from .meshnet import MeshNetLandmarkRegressor

        values.pop("backbone")
        values.pop("target_faces")
        return MeshNetLandmarkRegressor(**values)
    return build_landmark_model(values)
