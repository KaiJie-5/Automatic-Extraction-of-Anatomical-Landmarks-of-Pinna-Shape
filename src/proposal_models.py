"""Proposal-aligned locator, contour heads, and local landmark refinement."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .curve import contour_ids, validate_landmark_arc_fractions
from .pointnet2_model import PointNet2FeatureEncoder, PointNet2LandmarkRegressor
from .pointnet2_utils import index_points, square_distance
from .pointnext_model import PointNeXtEncoder


CONTOUR_LENGTHS = (25, 30, 20, 10)
REFINEMENT_ANCHOR_MODES = ("raw", "nearest-surface-sample")
HEATMAP_REFINEMENT_MODES = ("geometry-offset", "feature-attention")
BILATERAL_MODES = ("none", "shared-latent", "landmark-cross-attention")


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


class FeatureAwareSurfaceRefinementStage(nn.Module):
    """One landmark-conditioned attention step over nearby surface samples."""

    def __init__(
        self, feature_dim: int, hidden_dim: int, update_query: bool
    ):
        super().__init__()
        self.point_projection = nn.Linear(feature_dim, hidden_dim)
        self.geometry_projection = nn.Linear(7, hidden_dim)
        self.heatmap_projection = nn.Linear(2, hidden_dim)
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.query_update = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            if update_query
            else None
        )
        # Begin as a local re-decoding of the trained heatmap.  The learned
        # residual then specializes without an arbitrary initial displacement.
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)

    def forward(
        self,
        current: torch.Tensor,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        point_features: torch.Tensor,
        heatmap_logits: torch.Tensor,
        heatmap_entropy: torch.Tensor,
        query_state: torch.Tensor,
        k: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.cdist(current.float(), xyz.float()).topk(
            min(int(k), int(xyz.shape[1])), dim=-1, largest=False, sorted=True
        ).indices
        neighbour_xyz = index_points(xyz, indices).float()
        neighbour_normals = index_points(normals, indices).float()
        neighbour_features = index_points(point_features, indices)
        relative = neighbour_xyz - current.float().unsqueeze(2)
        distance = torch.linalg.norm(relative, dim=-1, keepdim=True)
        geometry = torch.cat([relative, neighbour_normals, distance], dim=-1)
        local_logits = torch.gather(heatmap_logits.float(), 2, indices)
        centred_logits = local_logits - local_logits.amax(dim=-1, keepdim=True)
        entropy = heatmap_entropy.unsqueeze(-1).expand_as(centred_logits)
        heatmap_values = torch.stack([centred_logits, entropy], dim=-1)

        point_hidden = self.point_projection(neighbour_features)
        geometry_hidden = self.geometry_projection(geometry)
        heatmap_hidden = self.heatmap_projection(heatmap_values)
        query_hidden = query_state.unsqueeze(2)
        candidates = self.candidate_norm(
            point_hidden
            + geometry_hidden.to(point_hidden.dtype)
            + heatmap_hidden.to(point_hidden.dtype)
            + query_hidden.to(point_hidden.dtype)
        )
        learned_residual = self.score_head(torch.nn.functional.gelu(candidates)).squeeze(-1)
        scores = centred_logits + learned_residual.float()
        weights = torch.softmax(scores / float(temperature), dim=-1)
        updated = (neighbour_xyz * weights.unsqueeze(-1)).sum(dim=2)
        context = (candidates.float() * weights.unsqueeze(-1)).sum(dim=2)
        next_query = query_state.float()
        if self.query_update is not None:
            next_query = self.query_update(
                torch.cat([query_state.float(), context], dim=-1)
            )
        return updated.float(), next_query.float()


class FeatureAwareSurfaceRefiner(nn.Module):
    """Iteratively refine landmarks using decoded features and heatmap confidence."""

    def __init__(
        self,
        k: int,
        feature_dim: int,
        stages: int = 2,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        if k not in {32, 64}:
            raise ValueError("feature-aware surface refinement k must be 32 or 64")
        if int(stages) not in {1, 2}:
            raise ValueError("feature-aware surface refinement stages must be 1 or 2")
        if int(hidden_dim) <= 0:
            raise ValueError("feature-aware surface refinement hidden_dim must be positive")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("feature-aware surface refinement temperature must be positive")
        self.k = int(k)
        self.stage_count = int(stages)
        self.temperature = float(temperature)
        self.query_projection = nn.Linear(int(feature_dim), int(hidden_dim))
        self.identity = nn.Embedding(85, int(hidden_dim))
        self.contour_identity = nn.Embedding(4, int(hidden_dim))
        contour_ids = torch.repeat_interleave(
            torch.arange(4), torch.as_tensor(CONTOUR_LENGTHS)
        )
        self.register_buffer("contour_ids", contour_ids, persistent=False)
        self.stages = nn.ModuleList(
            [
                FeatureAwareSurfaceRefinementStage(
                    feature_dim,
                    hidden_dim,
                    update_query=index < self.stage_count - 1,
                )
                for index in range(self.stage_count)
            ]
        )

    @staticmethod
    def _normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        probabilities = torch.exp(log_probabilities)
        denominator = max(math.log(max(int(logits.shape[-1]), 2)), 1.0)
        return -(
            probabilities * log_probabilities
        ).sum(dim=-1) / denominator

    def forward(
        self,
        coarse: torch.Tensor,
        point_cloud: torch.Tensor,
        point_features: torch.Tensor,
        heatmap_logits: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point_cloud.shape[-1] != 6:
            point_cloud = point_cloud.transpose(1, 2)
        xyz = point_cloud[..., :3].float()
        normals = point_cloud[..., 3:6].float()
        if query_features.ndim == 2:
            query_features = query_features.unsqueeze(0).expand(
                coarse.shape[0], -1, -1
            )
        identities = self.identity(
            torch.arange(85, device=coarse.device)
        ).unsqueeze(0)
        contours = self.contour_identity(self.contour_ids).unsqueeze(0)
        query_state = (
            self.query_projection(query_features)
            + identities
            + contours
        ).float()
        entropy = self._normalized_entropy(heatmap_logits)
        current = coarse.float()
        predictions = []
        for stage in self.stages:
            current, query_state = stage(
                current,
                xyz,
                normals,
                point_features,
                heatmap_logits,
                entropy,
                query_state,
                self.k,
                self.temperature,
            )
            predictions.append(current)
        return current, torch.stack(predictions, dim=1)


def _landmark_cascade_metadata() -> torch.Tensor:
    """Return fixed contour identity, position, and section-anchor metadata."""

    metadata = torch.zeros(85, 6, dtype=torch.float32)
    section_anchors = {
        0, 6, 22, 24, 25, 33, 42, 46, 50, 54, 55, 64, 74, 75, 84
    }
    start = 0
    for contour, length in enumerate(CONTOUR_LENGTHS):
        stop = start + int(length)
        metadata[start:stop, contour] = 1.0
        if length > 1:
            metadata[start:stop, 4] = torch.linspace(0.0, 1.0, int(length))
        start = stop
    metadata[list(sorted(section_anchors)), 5] = 1.0
    return metadata


def _landmark_cascade_attention_bias() -> torch.Tensor:
    """Softly favour same-contour neighbours without forbidding global context."""

    metadata = _landmark_cascade_metadata()
    contour = metadata[:, :4].argmax(dim=1)
    position = metadata[:, 4]
    same_contour = contour[:, None] == contour[None, :]
    within_contour = -2.0 * torch.abs(position[:, None] - position[None, :])
    bias = torch.full((85, 85), -2.5, dtype=torch.float32)
    bias = torch.where(same_contour, within_contour, bias)
    bias.fill_diagonal_(0.0)
    return bias


class SurfaceHeatmapLandmarkCascadeStage(nn.Module):
    """One full-surface, landmark-token residual heatmap refinement stage.

    The previous heatmap and coordinate define a soft spatial prior, never a
    hard crop, so a poor coarse hypothesis cannot remove the correct surface
    region. Landmark self-attention communicates contour context before a
    zero-initialized residual updates all 85 surface heatmaps.
    """

    def __init__(
        self,
        feature_dim: int,
        attention_heads: int,
        radius_normalized: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        attention_heads = int(attention_heads)
        radius_normalized = float(radius_normalized)
        dropout = float(dropout)
        if feature_dim <= 0 or attention_heads <= 0:
            raise ValueError("cascade feature width and attention heads must be positive")
        if feature_dim % attention_heads:
            raise ValueError("cascade attention heads must divide heatmap_feature_dim")
        if not math.isfinite(radius_normalized) or radius_normalized <= 0.0:
            raise ValueError("cascade radius must be positive and finite")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("cascade dropout must be finite and in [0, 1)")

        self.radius_normalized = radius_normalized
        self.logit_scale = feature_dim ** -0.5
        self.coordinate_projection = nn.Linear(3, feature_dim)
        self.uncertainty_projection = nn.Linear(2, feature_dim)
        self.metadata_projection = nn.Linear(6, feature_dim)
        self.surface_context_projection = nn.Linear(feature_dim, feature_dim)
        self.cross_norm = nn.LayerNorm(feature_dim)
        self.cross_feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.cross_output_norm = nn.LayerNorm(feature_dim)
        self.landmark_attention = nn.MultiheadAttention(
            feature_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.landmark_attention_norm = nn.LayerNorm(feature_dim)
        self.landmark_feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.landmark_output_norm = nn.LayerNorm(feature_dim)
        self.residual_query_projection = nn.Linear(feature_dim, feature_dim)
        # This makes Stage 0 and the cascaded model exactly equal before the
        # first update instead of introducing a random initial displacement.
        nn.init.zeros_(self.residual_query_projection.weight)
        nn.init.zeros_(self.residual_query_projection.bias)
        self.register_buffer(
            "landmark_metadata",
            _landmark_cascade_metadata(),
            persistent=False,
        )
        self.register_buffer(
            "landmark_attention_bias",
            _landmark_cascade_attention_bias(),
            persistent=False,
        )

    @staticmethod
    def _uncertainty(logits: torch.Tensor) -> torch.Tensor:
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        probabilities = torch.exp(log_probabilities)
        confidence = probabilities.amax(dim=-1)
        denominator = max(math.log(max(int(logits.shape[-1]), 2)), 1.0)
        entropy = -(probabilities * log_probabilities).sum(dim=-1) / denominator
        return torch.stack([confidence, entropy], dim=-1)

    def forward(
        self,
        previous_logits: torch.Tensor,
        current_coordinates: torch.Tensor,
        xyz: torch.Tensor,
        point_features: torch.Tensor,
        query_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_state.ndim == 2:
            query_state = query_state.unsqueeze(0).expand(
                previous_logits.shape[0], -1, -1
            )
        if query_state.ndim != 3 or query_state.shape[1] != 85:
            raise ValueError("cascade landmark tokens must have shape (B, 85, D)")

        # Every surface candidate remains available. The Gaussian term only
        # biases attention towards the current hypothesis.
        squared_distance = torch.cdist(
            current_coordinates.float(), xyz.float()
        ).square()
        spatial_bias = -squared_distance / (
            2.0 * self.radius_normalized * self.radius_normalized
        )
        surface_attention = torch.softmax(
            previous_logits.float() + spatial_bias, dim=-1
        )
        surface_context = torch.einsum(
            "bln,bnd->bld", surface_attention, point_features.float()
        )
        metadata = self.landmark_metadata.to(
            device=query_state.device, dtype=query_state.dtype
        ).unsqueeze(0)
        combined = (
            query_state
            + self.coordinate_projection(current_coordinates.to(query_state.dtype))
            + self.uncertainty_projection(
                self._uncertainty(previous_logits).to(query_state.dtype)
            )
            + self.metadata_projection(metadata)
            + self.surface_context_projection(surface_context.to(query_state.dtype))
        )
        hidden = self.cross_norm(combined)
        hidden = self.cross_output_norm(hidden + self.cross_feed_forward(hidden))
        attended, _ = self.landmark_attention(
            hidden,
            hidden,
            hidden,
            attn_mask=self.landmark_attention_bias.to(
                device=hidden.device, dtype=hidden.dtype
            ),
            need_weights=False,
        )
        hidden = self.landmark_attention_norm(hidden + attended)
        next_query = self.landmark_output_norm(
            hidden + self.landmark_feed_forward(hidden)
        )
        residual_query = self.residual_query_projection(next_query)
        residual_logits = torch.einsum(
            "bnd,bld->bln", point_features, residual_query
        ) * self.logit_scale
        return previous_logits.float() + residual_logits.float(), next_query


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
    before optional geometry-offset or feature-aware surface refinement. Four
    independent query branches preserve the official contour ordering when
    ``four_heads`` is enabled.
    """

    def __init__(
        self,
        encoder_config: Mapping[str, object] | None = None,
        four_heads: bool = True,
        heatmap_feature_dim: int = 128,
        heatmap_topk: int = 64,
        heatmap_coordinate_temperature: float = 1.0,
        surface_voting: bool = False,
        vote_cap_normalized: float = 0.0,
        vote_fusion_iterations: int = 3,
        vote_fusion_epsilon_normalized: float = 0.0,
        curve_enabled: bool = False,
        curve_landmark_fractions: Sequence[float] | None = None,
        curve_logit_weight: float = 0.5,
        curve_arc_logit_weight: float = 0.25,
        curve_arc_temperature: float = 0.1,
        refinement_k: int = 0,
        refinement_cap_normalized: float = 0.0,
        refinement_anchor: str = "raw",
        refinement_mode: str = "geometry-offset",
        refinement_stages: int = 1,
        refinement_hidden_dim: int = 128,
        refinement_temperature: float = 1.0,
        cascade_stages: int = 0,
        cascade_attention_heads: int = 8,
        cascade_radius_normalized: float = 0.0,
        cascade_radius_decay: float = 0.5,
        cascade_dropout: float = 0.0,
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
        refinement_mode = str(refinement_mode).lower()
        if refinement_mode not in HEATMAP_REFINEMENT_MODES:
            raise ValueError(
                "refinement_mode must be one of "
                f"{HEATMAP_REFINEMENT_MODES}"
            )
        refinement_stages = int(refinement_stages)
        if refinement_mode == "geometry-offset" and refinement_stages != 1:
            raise ValueError("geometry-offset refinement supports exactly one stage")
        if refinement_mode == "feature-attention":
            if not refinement_k:
                raise ValueError(
                    "feature-attention refinement requires refinement_k"
                )
            if refinement_anchor != "raw":
                raise ValueError(
                    "feature-attention refinement requires refinement_anchor='raw'"
                )
        self.encoder = PointNeXtEncoder(**dict(encoder_config or {}))
        channels = tuple(int(value) for value in self.encoder.stage_channels)
        if len(channels) != 5:
            raise ValueError("surface heatmap decoder requires five PointNeXt levels")
        self.four_heads = bool(four_heads)
        self.heatmap_topk = heatmap_topk
        self.heatmap_coordinate_temperature = heatmap_coordinate_temperature
        self.surface_voting = bool(surface_voting)
        self.vote_cap_normalized = float(vote_cap_normalized)
        self.vote_fusion_iterations = int(vote_fusion_iterations)
        self.vote_fusion_epsilon_normalized = float(
            vote_fusion_epsilon_normalized
        )
        if self.surface_voting and (
            not math.isfinite(self.vote_cap_normalized)
            or self.vote_cap_normalized <= 0.0
        ):
            raise ValueError(
                "surface voting requires a positive finite vote cap"
            )
        if not self.surface_voting and self.vote_cap_normalized != 0.0:
            raise ValueError("vote cap is only valid when surface voting is enabled")
        if self.surface_voting and (
            self.vote_fusion_iterations <= 0
            or not math.isfinite(self.vote_fusion_epsilon_normalized)
            or self.vote_fusion_epsilon_normalized <= 0.0
        ):
            raise ValueError(
                "surface voting requires positive robust-fusion settings"
            )
        self.refinement_mode = refinement_mode
        self.curve_enabled = bool(curve_enabled)
        cascade_stages = int(cascade_stages)
        cascade_attention_heads = int(cascade_attention_heads)
        cascade_radius_normalized = float(cascade_radius_normalized)
        cascade_radius_decay = float(cascade_radius_decay)
        cascade_dropout = float(cascade_dropout)
        if cascade_stages not in {0, 1, 2}:
            raise ValueError("surface heatmap cascade supports zero, one, or two stages")
        if cascade_stages and self.curve_enabled:
            raise ValueError(
                "landmark-token cascade and continuous curve decoding are separate experiments"
            )
        if cascade_stages and refinement_mode != "geometry-offset":
            raise ValueError(
                "landmark-token cascade requires geometry-offset final refinement"
            )
        if cascade_stages and (
            cascade_attention_heads <= 0
            or heatmap_feature_dim % cascade_attention_heads
        ):
            raise ValueError(
                "cascade attention heads must divide heatmap_feature_dim"
            )
        if cascade_stages and (
            not math.isfinite(cascade_radius_normalized)
            or cascade_radius_normalized <= 0.0
        ):
            raise ValueError("landmark-token cascade requires a positive radius")
        if (
            not math.isfinite(cascade_radius_decay)
            or not 0.0 < cascade_radius_decay <= 1.0
        ):
            raise ValueError("cascade radius decay must be in (0, 1]")
        if not math.isfinite(cascade_dropout) or not 0.0 <= cascade_dropout < 1.0:
            raise ValueError("cascade dropout must be finite and in [0, 1)")
        self.cascade_stage_count = cascade_stages
        self.curve_logit_weight = float(curve_logit_weight)
        self.curve_arc_logit_weight = float(curve_arc_logit_weight)
        self.curve_arc_temperature = float(curve_arc_temperature)
        if self.curve_enabled:
            if self.surface_voting:
                raise ValueError(
                    "continuous curve decoding and surface voting are separate experiments"
                )
            if curve_landmark_fractions is None:
                raise ValueError(
                    "continuous curve decoding requires training-fold landmark fractions"
                )
            fractions = validate_landmark_arc_fractions(
                curve_landmark_fractions
            )
            if (
                not math.isfinite(self.curve_logit_weight)
                or self.curve_logit_weight < 0.0
                or not math.isfinite(self.curve_arc_logit_weight)
                or self.curve_arc_logit_weight < 0.0
                or not math.isfinite(self.curve_arc_temperature)
                or self.curve_arc_temperature <= 0.0
            ):
                raise ValueError(
                    "continuous curve decoder weights must be non-negative and "
                    "its arc temperature must be positive"
                )
            self.register_buffer(
                "curve_landmark_fractions",
                torch.from_numpy(fractions),
            )
            self.register_buffer(
                "curve_contour_ids",
                torch.from_numpy(contour_ids()),
                persistent=False,
            )
        else:
            if curve_landmark_fractions is not None:
                raise ValueError(
                    "curve landmark fractions are only valid for curve decoding"
                )
            self.curve_landmark_fractions = None
            self.curve_contour_ids = None

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
        if self.curve_enabled:
            self.curve_field_head = nn.Linear(heatmap_feature_dim, 4)
            self.curve_arc_head = nn.Linear(heatmap_feature_dim, 4)
        else:
            self.curve_field_head = None
            self.curve_arc_head = None

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
        self.cascade_layers = nn.ModuleList(
            [
                SurfaceHeatmapLandmarkCascadeStage(
                    heatmap_feature_dim,
                    cascade_attention_heads,
                    cascade_radius_normalized * (cascade_radius_decay ** index),
                    cascade_dropout,
                )
                for index in range(cascade_stages)
            ]
        )
        if self.surface_voting:
            self.vote_query_projections = nn.ModuleList(
                [
                    nn.Linear(heatmap_feature_dim, heatmap_feature_dim)
                    for _ in range(3)
                ]
            )
            # Zero offsets start from the established surface candidates while
            # retaining gradients into the new vote heads.
            for projection in self.vote_query_projections:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
        else:
            self.vote_query_projections = None
        if not refinement_k:
            self.refiner = None
        elif refinement_mode == "geometry-offset":
            self.refiner = LocalLandmarkRefiner(
                refinement_k,
                refinement_cap_normalized,
                anchor_mode=refinement_anchor,
            )
        else:
            self.refiner = FeatureAwareSurfaceRefiner(
                refinement_k,
                heatmap_feature_dim,
                stages=refinement_stages,
                hidden_dim=refinement_hidden_dim,
                temperature=refinement_temperature,
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

    def _query_features(self) -> torch.Tensor:
        chunks = []
        for queries, projection in zip(
            self.query_embeddings, self.query_projections
        ):
            chunks.append(projection(queries))
        return torch.cat(chunks, dim=0)

    def _logits(
        self, point_features: torch.Tensor, query_features: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.einsum("bnd,ld->bln", point_features, query_features)
            * self.logit_scale
        )

    def _curve_fields(
        self, point_features: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.curve_field_head is None or self.curve_arc_head is None:
            return None, None
        curve_logits = self.curve_field_head(point_features).transpose(1, 2)
        arc_coordinates = torch.sigmoid(
            self.curve_arc_head(point_features).transpose(1, 2).float()
        )
        return curve_logits, arc_coordinates

    def _apply_curve_structure(
        self,
        unary_logits: torch.Tensor,
        curve_logits: torch.Tensor | None,
        arc_coordinates: torch.Tensor | None,
    ) -> torch.Tensor:
        """Intersect semantic landmark evidence with curve membership/order."""

        if not self.curve_enabled:
            return unary_logits
        if curve_logits is None or arc_coordinates is None:
            raise RuntimeError("continuous curve fields were not produced")
        contour_indices = self.curve_contour_ids.to(unary_logits.device)
        contour_log_probability = torch.log_softmax(
            curve_logits.float(), dim=-1
        ).index_select(1, contour_indices)
        landmark_arc = arc_coordinates.index_select(1, contour_indices)
        fractions = self.curve_landmark_fractions.to(
            device=unary_logits.device, dtype=torch.float32
        ).view(1, 85, 1)
        arc_penalty = -(
            (landmark_arc - fractions).square()
            / (2.0 * self.curve_arc_temperature**2)
        )
        return (
            unary_logits.float()
            + self.curve_logit_weight * contour_log_probability
            + self.curve_arc_logit_weight * arc_penalty
        )

    def _vote_offsets(
        self, point_features: torch.Tensor, query_features: torch.Tensor
    ) -> torch.Tensor | None:
        """Predict bounded candidate-to-landmark offsets in local coordinates."""
        if self.vote_query_projections is None:
            return None
        projected = torch.stack(
            [projection(query_features) for projection in self.vote_query_projections],
            dim=-2,
        )
        if projected.ndim == 3:
            raw = torch.einsum("bnd,lcd->blnc", point_features, projected)
        elif projected.ndim == 4:
            raw = torch.einsum("bnd,blcd->blnc", point_features, projected)
        else:
            raise ValueError("landmark query features have an invalid rank")
        raw = raw * self.logit_scale
        magnitude = torch.linalg.norm(raw, dim=-1, keepdim=True)
        bounded_scale = torch.where(
            magnitude > 1e-6,
            torch.tanh(magnitude) / magnitude.clamp_min(1e-6),
            torch.ones_like(magnitude),
        )
        return raw * bounded_scale * self.vote_cap_normalized

    def decode_surface_coordinates(
        self,
        logits: torch.Tensor,
        xyz: torch.Tensor,
        vote_offsets: torch.Tensor | None = None,
        topk: int | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        selected_topk = self.heatmap_topk if topk is None else int(topk)
        selected_temperature = (
            self.heatmap_coordinate_temperature
            if temperature is None
            else float(temperature)
        )
        if selected_topk <= 0:
            raise ValueError("surface heatmap top-k must be positive")
        if (
            not math.isfinite(selected_temperature)
            or selected_temperature <= 0.0
        ):
            raise ValueError("surface heatmap coordinate temperature must be positive")
        count = min(selected_topk, int(xyz.shape[1]))
        values, indices = logits.topk(count, dim=-1, largest=True, sorted=True)
        candidates = index_points(xyz, indices)
        if vote_offsets is not None:
            if vote_offsets.shape != (*logits.shape, 3):
                raise ValueError(
                    "surface vote offsets must have shape (B, 85, N, 3)"
                )
            candidates = candidates + torch.gather(
                vote_offsets,
                2,
                indices.unsqueeze(-1).expand(-1, -1, -1, 3),
            )
        weights = torch.softmax(
            values.float() / selected_temperature, dim=-1
        ).to(candidates.dtype)
        estimate = (candidates * weights.unsqueeze(-1)).sum(dim=2)
        if vote_offsets is not None:
            for _ in range(self.vote_fusion_iterations):
                residual = torch.linalg.norm(
                    candidates - estimate.unsqueeze(2), dim=-1
                )
                robust_weights = weights / residual.clamp_min(
                    self.vote_fusion_epsilon_normalized
                )
                robust_weights = robust_weights / robust_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                estimate = (
                    candidates * robust_weights.unsqueeze(-1)
                ).sum(dim=2)
        return estimate.float()

    def apply_refinement(
        self,
        coarse: torch.Tensor,
        points: torch.Tensor,
        point_features: torch.Tensor,
        logits: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.refiner is None:
            return coarse, None
        if self.refinement_mode == "feature-attention":
            return self.refiner(
                coarse,
                points,
                point_features,
                logits,
                query_features,
            )
        final = self.refiner(coarse, points)
        return final, final.unsqueeze(1)

    def forward_with_details(self, points: torch.Tensor) -> Mapping[str, torch.Tensor]:
        levels = self.encoder.forward_features(points)
        point_features = self._decode_point_features(levels)
        query_features = self._query_features()
        unary_logits = self._logits(point_features, query_features)
        curve_logits, curve_arc_coordinates = self._curve_fields(point_features)
        logits = self._apply_curve_structure(
            unary_logits, curve_logits, curve_arc_coordinates
        )
        xyz = levels["xyz"][0]
        initial_prediction = self.decode_surface_coordinates(logits, xyz)
        cascade_predictions = []
        cascade_logits = []
        current = initial_prediction
        current_query = query_features
        for cascade_layer in self.cascade_layers:
            cascade_logits.append(logits)
            cascade_predictions.append(current)
            logits, current_query = cascade_layer(
                logits,
                current,
                xyz,
                point_features,
                current_query,
            )
            current = self.decode_surface_coordinates(logits, xyz)
        vote_offsets = self._vote_offsets(point_features, current_query)
        coarse = (
            self.decode_surface_coordinates(logits, xyz, vote_offsets)
            if vote_offsets is not None
            else current
        )
        final, refinement_predictions = self.apply_refinement(
            coarse,
            points,
            point_features,
            logits,
            current_query,
        )
        details = {
            "coarse": coarse,
            "final": final,
            "heatmap_logits": logits,
            "unary_heatmap_logits": unary_logits,
            "surface_candidates": xyz,
            "decoded_point_features": point_features,
            "landmark_query_features": current_query,
        }
        if cascade_predictions:
            details["cascade_initial_prediction"] = initial_prediction
            details["cascade_aux_predictions"] = torch.stack(
                cascade_predictions, dim=1
            )
            details["cascade_aux_logits"] = torch.stack(cascade_logits, dim=1)
            details["cascade_stage_predictions"] = torch.stack(
                cascade_predictions[1:] + [coarse], dim=1
            )
        if refinement_predictions is not None:
            details["refinement_stage_predictions"] = refinement_predictions
        if vote_offsets is not None:
            details["surface_vote_offsets"] = vote_offsets
        if curve_logits is not None and curve_arc_coordinates is not None:
            details["curve_logits"] = curve_logits
            details["curve_arc_coordinates"] = curve_arc_coordinates
            details["curve_landmark_fractions"] = (
                self.curve_landmark_fractions
            )
        return details

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_with_details(points)["final"]


class BilateralLandmarkCrossAttentionBlock(nn.Module):
    """Symmetric cross-attention between left/right landmark token sets."""

    def __init__(self, feature_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if feature_dim % heads:
            raise ValueError(
                "bilateral attention heads must divide heatmap_feature_dim"
            )
        self.cross_attention = nn.MultiheadAttention(
            int(feature_dim),
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(int(feature_dim))
        self.feed_forward = nn.Sequential(
            nn.Linear(int(feature_dim), int(feature_dim) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(feature_dim) * 4, int(feature_dim)),
        )
        self.output_norm = nn.LayerNorm(int(feature_dim))

    def _update(
        self, query: torch.Tensor, opposite: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query,
            opposite,
            opposite,
            need_weights=False,
        )
        hidden = self.attention_norm(query + attended)
        return self.output_norm(hidden + self.feed_forward(hidden))

    def forward(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both updates read the same pre-update pair.  Reusing the block weights
        # makes the fusion symmetric rather than privileging one ear.
        return self._update(left, right), self._update(right, left)


class BilateralPointNeXtSurfaceHeatmapRegressor(nn.Module):
    """Paired-ear extension of the proven PointNeXt surface heatmap model.

    Input and output ear order is always ``(left, mirrored-right)``.  The
    underlying PointNeXt encoder, feature propagation, heatmap coordinate
    decoder, and local refiner are unchanged and weight-shared between ears.
    Only the landmark-query conditioning differs between bilateral modes.
    """

    def __init__(
        self,
        bilateral_mode: str,
        bilateral_attention_heads: int = 8,
        bilateral_attention_layers: int = 1,
        bilateral_dropout: float = 0.0,
        **single_ear_config,
    ):
        super().__init__()
        mode = str(bilateral_mode).lower()
        if mode not in BILATERAL_MODES[1:]:
            raise ValueError(
                "bilateral landmark model requires bilateral_mode to be "
                "'shared-latent' or 'landmark-cross-attention'"
            )
        attention_layers = int(bilateral_attention_layers)
        attention_heads = int(bilateral_attention_heads)
        dropout = float(bilateral_dropout)
        if attention_layers <= 0 or attention_heads <= 0:
            raise ValueError("bilateral attention layers and heads must be positive")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("bilateral dropout must be finite and in [0, 1)")

        self.bilateral_mode = mode
        self.ear_model = PointNeXtSurfaceHeatmapRegressor(
            **single_ear_config
        )
        feature_dim = int(single_ear_config.get("heatmap_feature_dim", 128))
        global_dim = int(self.ear_model.encoder.stage_channels[-1])
        self.feature_dim = feature_dim

        if mode == "shared-latent":
            # Mean and absolute difference are invariant to swapping the two
            # ears, while retaining both common morphology and asymmetry.
            self.subject_projection = nn.Sequential(
                nn.Linear(global_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
            self.query_norm = nn.LayerNorm(feature_dim)
            self.own_token_projection = None
            self.cross_attention_blocks = nn.ModuleList()
        else:
            if feature_dim % attention_heads:
                raise ValueError(
                    "--bilateral-attention-heads must divide "
                    "--heatmap-feature-dim"
                )
            self.subject_projection = None
            self.query_norm = nn.LayerNorm(feature_dim)
            self.own_token_projection = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
            )
            self.cross_attention_blocks = nn.ModuleList(
                [
                    BilateralLandmarkCrossAttentionBlock(
                        feature_dim, attention_heads, dropout
                    )
                    for _ in range(attention_layers)
                ]
            )

    @property
    def encoder(self):
        """Expose the shared backbone for the optional encoder learning rate."""
        return self.ear_model.encoder

    @staticmethod
    def _validate_points(points: torch.Tensor) -> tuple[int, int, int]:
        if points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 6:
            raise ValueError(
                "bilateral landmark input must have shape (B, 2, N, 6) "
                "in (left, mirrored-right) order"
            )
        return int(points.shape[0]), int(points.shape[1]), int(points.shape[2])

    def _condition_queries(
        self,
        point_features: torch.Tensor,
        ear_globals: torch.Tensor,
        base_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        batch = int(point_features.shape[0])
        if self.bilateral_mode == "shared-latent":
            mean = ear_globals.mean(dim=1)
            absolute_difference = torch.abs(
                ear_globals[:, 0] - ear_globals[:, 1]
            )
            subject_latent = self.subject_projection(
                torch.cat([mean, absolute_difference], dim=-1)
            )
            queries = self.query_norm(
                base_queries.view(1, 1, 85, -1)
                + subject_latent[:, None, None, :]
            ).expand(-1, 2, -1, -1)
            return queries, {"bilateral_subject_latent": subject_latent}

        base = base_queries.view(1, 1, 85, -1).expand(
            batch, 2, -1, -1
        )
        preliminary_logits = torch.einsum(
            "bend,ld->beln", point_features, base_queries
        ) * self.ear_model.logit_scale
        probabilities = torch.softmax(
            preliminary_logits.float(), dim=-1
        ).to(point_features.dtype)
        own_surface_tokens = torch.einsum(
            "beln,bend->beld", probabilities, point_features
        )
        tokens = self.query_norm(
            base + self.own_token_projection(own_surface_tokens)
        )
        left, right = tokens[:, 0], tokens[:, 1]
        for block in self.cross_attention_blocks:
            left, right = block(left, right)
        queries = torch.stack([left, right], dim=1)
        return queries, {
            "bilateral_preliminary_heatmap_logits": preliminary_logits,
            "bilateral_own_surface_tokens": own_surface_tokens,
        }

    def forward_with_details(
        self, points: torch.Tensor
    ) -> Mapping[str, torch.Tensor]:
        batch, ears, point_count = self._validate_points(points)
        flat_points = points.reshape(batch * ears, point_count, 6)
        levels = self.ear_model.encoder.forward_features(flat_points)
        flat_point_features = self.ear_model._decode_point_features(levels)
        point_features = flat_point_features.reshape(
            batch, ears, point_count, self.feature_dim
        )
        ear_globals = levels["global"].reshape(batch, ears, -1)
        base_queries = self.ear_model._query_features()
        query_features, bilateral_details = self._condition_queries(
            point_features, ear_globals, base_queries
        )
        logits = torch.einsum(
            "bend,beld->beln", point_features, query_features
        ) * self.ear_model.logit_scale
        xyz = levels["xyz"][0].reshape(batch, ears, point_count, 3)

        flat_logits = logits.reshape(batch * ears, 85, point_count)
        flat_xyz = xyz.reshape(batch * ears, point_count, 3)
        flat_queries = query_features.reshape(
            batch * ears, 85, self.feature_dim
        )
        vote_offsets = self.ear_model._vote_offsets(
            flat_point_features, flat_queries
        )
        coarse_flat = self.ear_model.decode_surface_coordinates(
            flat_logits, flat_xyz, vote_offsets
        )
        final_flat, refinement_predictions = self.ear_model.apply_refinement(
            coarse_flat,
            flat_points,
            flat_point_features,
            flat_logits,
            flat_queries,
        )
        details = {
            "coarse": coarse_flat.reshape(batch, ears, 85, 3),
            "final": final_flat.reshape(batch, ears, 85, 3),
            "heatmap_logits": logits,
            "surface_candidates": xyz,
            "decoded_point_features": point_features,
            "landmark_query_features": query_features,
            **bilateral_details,
        }
        if refinement_predictions is not None:
            details["refinement_stage_predictions"] = (
                refinement_predictions.reshape(
                    batch, ears, *refinement_predictions.shape[1:]
                )
            )
        if vote_offsets is not None:
            details["surface_vote_offsets"] = vote_offsets.reshape(
                batch, ears, 85, point_count, 3
            )
        return details

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
    bilateral_mode = str(values.pop("bilateral_mode", "none"))
    bilateral_attention_heads = int(
        values.pop("bilateral_attention_heads", 8)
    )
    bilateral_attention_layers = int(
        values.pop("bilateral_attention_layers", 1)
    )
    bilateral_dropout = float(values.pop("bilateral_dropout", 0.0))
    if decoder in {"surface_heatmap", "surface_curve"}:
        backbone = str(values.pop("backbone", ""))
        if backbone != "pointnext":
            raise ValueError("surface heatmap/curve decoder requires PointNeXt")
        if decoder == "surface_curve":
            values["curve_enabled"] = True
        if bilateral_mode != "none":
            if decoder == "surface_curve":
                raise ValueError(
                    "continuous curve decoding does not support bilateral neural modes"
                )
            if int(values.get("cascade_stages", 0)):
                raise ValueError(
                    "landmark-token cascade and bilateral neural modes are separate experiments"
                )
            return BilateralPointNeXtSurfaceHeatmapRegressor(
                bilateral_mode=bilateral_mode,
                bilateral_attention_heads=bilateral_attention_heads,
                bilateral_attention_layers=bilateral_attention_layers,
                bilateral_dropout=bilateral_dropout,
                **values,
            )
        return PointNeXtSurfaceHeatmapRegressor(**values)
    if bilateral_mode != "none":
        raise ValueError(
            "bilateral landmark modes require decoder='surface_heatmap'"
        )
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
