"""Adapter from fixed pinna samples to the official Point Transformer V3.

The official detached PTv3 implementation consumes flattened, voxelised point
dictionaries. This module keeps that implementation isolated behind the same
global-feature encoder contract used by the proposal landmark regressor.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn


PTV3_UPSTREAM_REPOSITORY = "https://github.com/Pointcept/PointTransformerV3"
PTV3_UPSTREAM_REVISION = "3229e9b7de1770c8ad17c316f8e349982de509f8"
PTV3_REQUIRED_MODULES = ("addict", "timm", "spconv", "torch_scatter", "flash_attn")


class PointTransformerV3DependencyError(ImportError):
    """Raised when the optional compiled PTv3 environment is unavailable."""


def _official_model_module():
    try:
        from .third_party.pointtransformerv3 import model as official_model
    except (ImportError, ModuleNotFoundError) as error:
        missing = getattr(error, "name", None) or str(error)
        raise PointTransformerV3DependencyError(
            "Point Transformer V3 requires the separate CUDA research environment "
            "with addict, timm, spconv, torch-scatter, and flash-attn. "
            "Create anthropometric_ptv3_env and install requirements-ptv3.txt; "
            f"the failing import was {missing!r}."
        ) from error
    if official_model.flash_attn is None:
        raise PointTransformerV3DependencyError(
            "Point Transformer V3 is configured to require FlashAttention, but "
            "flash_attn could not be imported. No non-Flash fallback is enabled."
        )
    return official_model


def _as_bnc(point_cloud: torch.Tensor, input_channels: int) -> torch.Tensor:
    if point_cloud.ndim != 3:
        raise ValueError("point_cloud must have shape (B, N, C) or (B, C, N)")
    if point_cloud.shape[-1] == input_channels:
        return point_cloud
    if point_cloud.shape[1] == input_channels:
        return point_cloud.transpose(1, 2)
    raise ValueError(f"expected {input_channels} input channels")


def deterministic_voxelize(
    point_cloud: torch.Tensor,
    grid_size: float,
    input_channels: int = 6,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Keep the first input sample in every occupied per-ear voxel."""

    points = _as_bnc(point_cloud, input_channels)
    if points.shape[1] <= 0:
        raise ValueError("point_cloud must contain at least one point per ear")
    if not torch.isfinite(points).all():
        raise ValueError("point_cloud contains non-finite values")
    if not 0.0 < float(grid_size) < 1.0:
        raise ValueError("PTv3 grid_size must be between zero and one in local coordinates")

    flat_features = []
    flat_coordinates = []
    flat_grid_coordinates = []
    flat_batches = []
    counts: list[int] = []
    for batch_index in range(points.shape[0]):
        values = points[batch_index]
        xyz = values[:, :3]
        grid = torch.floor(xyz.detach() / float(grid_size)).to(torch.long)
        grid = grid - grid.amin(dim=0, keepdim=True)
        unique_grid, inverse = torch.unique(
            grid, dim=0, sorted=True, return_inverse=True
        )
        source_indices = torch.arange(grid.shape[0], device=grid.device, dtype=torch.long)
        first_indices = torch.full(
            (unique_grid.shape[0],), grid.shape[0], device=grid.device, dtype=torch.long
        )
        first_indices.scatter_reduce_(
            0, inverse, source_indices, reduce="amin", include_self=True
        )
        selected_count = int(first_indices.numel())
        if selected_count <= 0:
            raise RuntimeError(f"voxelisation removed every point in batch item {batch_index}")
        flat_features.append(values[first_indices])
        flat_coordinates.append(xyz[first_indices])
        flat_grid_coordinates.append(unique_grid.to(torch.int32))
        flat_batches.append(
            torch.full(
                (selected_count,), batch_index, device=points.device, dtype=torch.long
            )
        )
        counts.append(selected_count)

    count_tensor = torch.as_tensor(counts, device=points.device, dtype=torch.long)
    data = {
        "feat": torch.cat(flat_features, dim=0).contiguous(),
        "coord": torch.cat(flat_coordinates, dim=0).contiguous(),
        "grid_coord": torch.cat(flat_grid_coordinates, dim=0).contiguous(),
        "batch": torch.cat(flat_batches, dim=0).contiguous(),
        "offset": torch.cumsum(count_tensor, dim=0),
    }
    return data, counts


class PointTransformerV3Encoder(nn.Module):
    """Full official PTv3 encoder-decoder followed by per-ear max pooling."""

    def __init__(
        self,
        input_channels: int = 6,
        grid_size: float = 0.01,
        order: Sequence[str] = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride: Sequence[int] = (2, 2, 2, 2),
        enc_depths: Sequence[int] = (2, 2, 2, 6, 2),
        enc_channels: Sequence[int] = (32, 64, 128, 256, 512),
        enc_num_head: Sequence[int] = (2, 4, 8, 16, 32),
        enc_patch_size: Sequence[int] = (1024, 1024, 1024, 1024, 1024),
        dec_depths: Sequence[int] = (2, 2, 2, 2),
        dec_channels: Sequence[int] = (64, 64, 128, 256),
        dec_num_head: Sequence[int] = (4, 4, 8, 16),
        dec_patch_size: Sequence[int] = (1024, 1024, 1024, 1024),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.3,
        pre_norm: bool = True,
        enable_rpe: bool = False,
        enable_flash: bool = True,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        global_pool: str = "max",
        voxel_representative: str = "first_input_index",
        order_shuffle_policy: str = "training_only",
        upstream_revision: str = PTV3_UPSTREAM_REVISION,
    ):
        super().__init__()
        if upstream_revision != PTV3_UPSTREAM_REVISION:
            raise ValueError(
                f"unsupported PTv3 revision {upstream_revision!r}; "
                f"expected {PTV3_UPSTREAM_REVISION}"
            )
        if not enable_flash:
            raise ValueError("the PTv3 research backbone requires FlashAttention")
        if global_pool != "max":
            raise ValueError("the PTv3 adapter supports only global max pooling")
        if voxel_representative != "first_input_index":
            raise ValueError("the PTv3 adapter requires deterministic first-point voxels")
        if order_shuffle_policy != "training_only":
            raise ValueError("the PTv3 adapter requires training-only order shuffling")
        if len(dec_channels) != 4 or int(dec_channels[0]) <= 0:
            raise ValueError("full PTv3 decoding requires four positive decoder channel values")

        official_model = _official_model_module()
        self.input_channels = int(input_channels)
        self.grid_size = float(grid_size)
        self.global_pool = global_pool
        self.voxel_representative = voxel_representative
        self.order_shuffle_policy = order_shuffle_policy
        self.upstream_revision = upstream_revision
        self.feature_dim = int(dec_channels[0])
        self.last_voxel_counts: tuple[int, ...] = ()
        self.model = official_model.PointTransformerV3(
            in_channels=self.input_channels,
            order=tuple(order),
            stride=tuple(stride),
            enc_depths=tuple(enc_depths),
            enc_channels=tuple(enc_channels),
            enc_num_head=tuple(enc_num_head),
            enc_patch_size=tuple(enc_patch_size),
            dec_depths=tuple(dec_depths),
            dec_channels=tuple(dec_channels),
            dec_num_head=tuple(dec_num_head),
            dec_patch_size=tuple(dec_patch_size),
            mlp_ratio=float(mlp_ratio),
            qkv_bias=bool(qkv_bias),
            qk_scale=qk_scale,
            attn_drop=float(attn_drop),
            proj_drop=float(proj_drop),
            drop_path=float(drop_path),
            pre_norm=bool(pre_norm),
            shuffle_orders=True,
            enable_rpe=bool(enable_rpe),
            enable_flash=True,
            upcast_attention=bool(upcast_attention),
            upcast_softmax=bool(upcast_softmax),
            cls_mode=False,
        )

    def _set_order_shuffle(self) -> None:
        enabled = bool(self.training)
        for module in self.model.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = enabled

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        points = _as_bnc(point_cloud, self.input_channels)
        if points.device.type != "cuda":
            raise RuntimeError(
                "the exact official Point Transformer V3 research backbone requires a CUDA device"
            )
        data, counts = deterministic_voxelize(
            points, grid_size=self.grid_size, input_channels=self.input_channels
        )
        self.last_voxel_counts = tuple(counts)
        self._set_order_shuffle()
        decoded = self.model(data)
        features = decoded.feat
        batch = decoded.batch.long()
        pooled = features.new_full(
            (points.shape[0], features.shape[-1]), -torch.inf
        )
        pooled.scatter_reduce_(
            0,
            batch.unsqueeze(-1).expand(-1, features.shape[-1]),
            features,
            reduce="amax",
            include_self=True,
        )
        if not torch.isfinite(pooled).all():
            raise RuntimeError("PTv3 global features contain non-finite values")
        return pooled


def default_pointtransformerv3_config(grid_size: float = 0.01) -> Mapping[str, object]:
    """Exact full encoder-decoder settings used by the controlled experiment."""

    return {
        "input_channels": 6,
        "grid_size": float(grid_size),
        "order": ["z", "z-trans", "hilbert", "hilbert-trans"],
        "stride": [2, 2, 2, 2],
        "enc_depths": [2, 2, 2, 6, 2],
        "enc_channels": [32, 64, 128, 256, 512],
        "enc_num_head": [2, 4, 8, 16, 32],
        "enc_patch_size": [1024, 1024, 1024, 1024, 1024],
        "dec_depths": [2, 2, 2, 2],
        "dec_channels": [64, 64, 128, 256],
        "dec_num_head": [4, 4, 8, 16],
        "dec_patch_size": [1024, 1024, 1024, 1024],
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "drop_path": 0.3,
        "pre_norm": True,
        "enable_rpe": False,
        "enable_flash": True,
        "upcast_attention": False,
        "upcast_softmax": False,
        "global_pool": "max",
        "voxel_representative": "first_input_index",
        "order_shuffle_policy": "training_only",
        "upstream_revision": PTV3_UPSTREAM_REVISION,
    }
