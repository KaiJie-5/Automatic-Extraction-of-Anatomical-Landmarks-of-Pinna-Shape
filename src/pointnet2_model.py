"""PointNet++ landmark regression model."""

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

from .pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg


def _as_list(values: Sequence[int]) -> List[int]:
    return [int(value) for value in values]


def _as_nested_ints(values: Sequence[Sequence[int]]) -> List[List[int]]:
    return [[int(item) for item in group] for group in values]


def _as_nested_floats(values: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(item) for item in group] for group in values]


class PointNet2LandmarkRegressor(nn.Module):
    """PointNet++ feature extractor with an MLP regression head.

    Args:
        num_landmarks: Number of 3D landmarks to regress. The challenge target is 170.
        input_channels: Number of point features. Use 6 for xyz + normals.
        use_normals: Whether channels 3:6 should be passed as point features.
        variant: "ssg" or "msg".
        head_channels: Hidden sizes for the final MLP regression head.
        dropout: Dropout probability used in the regression head.
    """

    def __init__(
        self,
        num_landmarks: int = 170,
        input_channels: int = 6,
        use_normals: bool = True,
        variant: str = "ssg",
        head_channels: Sequence[int] = (512, 256),
        dropout: float = 0.4,
        ssg_npoints: Sequence[int] = (512, 128),
        ssg_radii: Sequence[float] = (0.2, 0.4),
        ssg_nsamples: Sequence[int] = (32, 64),
        ssg_mlps: Sequence[Sequence[int]] = (
            (64, 64, 128),
            (128, 128, 256),
            (256, 512, 1024),
        ),
        msg_npoints: Sequence[int] = (512, 128),
        msg_radii: Sequence[Sequence[float]] = (
            (0.1, 0.2, 0.4),
            (0.2, 0.4, 0.8),
        ),
        msg_nsamples: Sequence[Sequence[int]] = (
            (16, 32, 128),
            (32, 64, 128),
        ),
        msg_mlps: Sequence[Sequence[Sequence[int]]] = (
            ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
            ((64, 64, 128), (128, 128, 256), (128, 128, 256)),
        ),
        msg_global_mlp: Sequence[int] = (256, 512, 1024),
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must include at least xyz coordinates")
        if use_normals and input_channels < 6:
            raise ValueError("use_normals=True requires input_channels >= 6")

        self.num_landmarks = int(num_landmarks)
        self.input_channels = int(input_channels)
        self.use_normals = bool(use_normals)
        self.variant = variant.lower()

        if self.variant == "ssg":
            global_feature_dim = self._build_ssg(
                ssg_npoints=ssg_npoints,
                ssg_radii=ssg_radii,
                ssg_nsamples=ssg_nsamples,
                ssg_mlps=ssg_mlps,
            )
        elif self.variant == "msg":
            global_feature_dim = self._build_msg(
                msg_npoints=msg_npoints,
                msg_radii=msg_radii,
                msg_nsamples=msg_nsamples,
                msg_mlps=msg_mlps,
                msg_global_mlp=msg_global_mlp,
            )
        else:
            raise ValueError(f"Unsupported PointNet++ variant: {variant}")

        self.regression_head = self._build_regression_head(
            input_dim=global_feature_dim,
            hidden_dims=_as_list(head_channels),
            output_dim=self.num_landmarks * 3,
            dropout=float(dropout),
        )

    def _build_ssg(
        self,
        ssg_npoints: Sequence[int],
        ssg_radii: Sequence[float],
        ssg_nsamples: Sequence[int],
        ssg_mlps: Sequence[Sequence[int]],
    ) -> int:
        npoints = _as_list(ssg_npoints)
        radii = [float(value) for value in ssg_radii]
        nsamples = _as_list(ssg_nsamples)
        mlps = _as_nested_ints(ssg_mlps)
        if len(npoints) != 2 or len(radii) != 2 or len(nsamples) != 2 or len(mlps) != 3:
            raise ValueError("SSG config requires 2 local layers and 1 global MLP")

        first_in = 6 if self.use_normals else 3
        self.sa1 = PointNetSetAbstraction(
            npoint=npoints[0],
            radius=radii[0],
            nsample=nsamples[0],
            in_channel=first_in,
            mlp=mlps[0],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=npoints[1],
            radius=radii[1],
            nsample=nsamples[1],
            in_channel=mlps[0][-1] + 3,
            mlp=mlps[1],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=mlps[1][-1] + 3,
            mlp=mlps[2],
            group_all=True,
        )
        return mlps[2][-1]

    def _build_msg(
        self,
        msg_npoints: Sequence[int],
        msg_radii: Sequence[Sequence[float]],
        msg_nsamples: Sequence[Sequence[int]],
        msg_mlps: Sequence[Sequence[Sequence[int]]],
        msg_global_mlp: Sequence[int],
    ) -> int:
        npoints = _as_list(msg_npoints)
        radii = _as_nested_floats(msg_radii)
        nsamples = _as_nested_ints(msg_nsamples)
        mlps = [[_as_list(scale) for scale in layer] for layer in msg_mlps]
        global_mlp = _as_list(msg_global_mlp)
        if len(npoints) != 2 or len(radii) != 2 or len(nsamples) != 2 or len(mlps) != 2:
            raise ValueError("MSG config requires 2 multi-scale layers")

        first_feature_dim = 3 if self.use_normals else 0
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=npoints[0],
            radius_list=radii[0],
            nsample_list=nsamples[0],
            in_channel=first_feature_dim,
            mlp_list=mlps[0],
        )
        sa1_out = sum(scale[-1] for scale in mlps[0])
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=npoints[1],
            radius_list=radii[1],
            nsample_list=nsamples[1],
            in_channel=sa1_out,
            mlp_list=mlps[1],
        )
        sa2_out = sum(scale[-1] for scale in mlps[1])
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=sa2_out + 3,
            mlp=global_mlp,
            group_all=True,
        )
        return global_mlp[-1]

    @staticmethod
    def _build_regression_head(
        input_dim: int, hidden_dims: Iterable[int], output_dim: int, dropout: float
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Regress landmarks from a batch of point clouds.

        Args:
            point_cloud: Shape (B, N, C) or (B, C, N). C must include xyz first.

        Returns:
            Normalized landmark coordinates with shape (B, num_landmarks, 3).
        """
        if point_cloud.ndim != 3:
            raise ValueError("point_cloud must have shape (B, N, C) or (B, C, N)")
        if point_cloud.shape[-1] == self.input_channels:
            point_cloud = point_cloud.permute(0, 2, 1)
        elif point_cloud.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected input_channels={self.input_channels}, got shape {tuple(point_cloud.shape)}"
            )

        xyz = point_cloud[:, :3, :]
        normals = point_cloud[:, 3:6, :] if self.use_normals else None
        l1_xyz, l1_points = self.sa1(xyz, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        global_features = l3_points.squeeze(-1)
        landmarks = self.regression_head(global_features)
        return landmarks.view(point_cloud.shape[0], self.num_landmarks, 3)


class PointNet2FeatureEncoder(nn.Module):
    """Reusable PointNet++ encoder that returns one global feature vector."""

    def __init__(
        self,
        input_channels: int = 6,
        use_normals: bool = True,
        variant: str = "ssg",
        ssg_npoints: Sequence[int] = (512, 128),
        ssg_radii: Sequence[float] = (0.2, 0.4),
        ssg_nsamples: Sequence[int] = (32, 64),
        ssg_mlps: Sequence[Sequence[int]] = (
            (64, 64, 128),
            (128, 128, 256),
            (256, 512, 1024),
        ),
        msg_npoints: Sequence[int] = (512, 128),
        msg_radii: Sequence[Sequence[float]] = (
            (0.1, 0.2, 0.4),
            (0.2, 0.4, 0.8),
        ),
        msg_nsamples: Sequence[Sequence[int]] = (
            (16, 32, 128),
            (32, 64, 128),
        ),
        msg_mlps: Sequence[Sequence[Sequence[int]]] = (
            ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
            ((64, 64, 128), (128, 128, 256), (128, 128, 256)),
        ),
        msg_global_mlp: Sequence[int] = (256, 512, 1024),
    ):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must include at least xyz coordinates")
        if use_normals and input_channels < 6:
            raise ValueError("use_normals=True requires input_channels >= 6")

        self.input_channels = int(input_channels)
        self.use_normals = bool(use_normals)
        self.variant = variant.lower()

        if self.variant == "ssg":
            self.feature_dim = self._build_ssg(
                ssg_npoints=ssg_npoints,
                ssg_radii=ssg_radii,
                ssg_nsamples=ssg_nsamples,
                ssg_mlps=ssg_mlps,
            )
        elif self.variant == "msg":
            self.feature_dim = self._build_msg(
                msg_npoints=msg_npoints,
                msg_radii=msg_radii,
                msg_nsamples=msg_nsamples,
                msg_mlps=msg_mlps,
                msg_global_mlp=msg_global_mlp,
            )
        else:
            raise ValueError(f"Unsupported PointNet++ variant: {variant}")

    def _build_ssg(
        self,
        ssg_npoints: Sequence[int],
        ssg_radii: Sequence[float],
        ssg_nsamples: Sequence[int],
        ssg_mlps: Sequence[Sequence[int]],
    ) -> int:
        npoints = _as_list(ssg_npoints)
        radii = [float(value) for value in ssg_radii]
        nsamples = _as_list(ssg_nsamples)
        mlps = _as_nested_ints(ssg_mlps)
        if len(npoints) != 2 or len(radii) != 2 or len(nsamples) != 2 or len(mlps) != 3:
            raise ValueError("SSG config requires 2 local layers and 1 global MLP")

        first_in = 6 if self.use_normals else 3
        self.sa1 = PointNetSetAbstraction(
            npoint=npoints[0],
            radius=radii[0],
            nsample=nsamples[0],
            in_channel=first_in,
            mlp=mlps[0],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=npoints[1],
            radius=radii[1],
            nsample=nsamples[1],
            in_channel=mlps[0][-1] + 3,
            mlp=mlps[1],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=mlps[1][-1] + 3,
            mlp=mlps[2],
            group_all=True,
        )
        return mlps[2][-1]

    def _build_msg(
        self,
        msg_npoints: Sequence[int],
        msg_radii: Sequence[Sequence[float]],
        msg_nsamples: Sequence[Sequence[int]],
        msg_mlps: Sequence[Sequence[Sequence[int]]],
        msg_global_mlp: Sequence[int],
    ) -> int:
        npoints = _as_list(msg_npoints)
        radii = _as_nested_floats(msg_radii)
        nsamples = _as_nested_ints(msg_nsamples)
        mlps = [[_as_list(scale) for scale in layer] for layer in msg_mlps]
        global_mlp = _as_list(msg_global_mlp)
        if len(npoints) != 2 or len(radii) != 2 or len(nsamples) != 2 or len(mlps) != 2:
            raise ValueError("MSG config requires 2 multi-scale layers")

        first_feature_dim = 3 if self.use_normals else 0
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=npoints[0],
            radius_list=radii[0],
            nsample_list=nsamples[0],
            in_channel=first_feature_dim,
            mlp_list=mlps[0],
        )
        sa1_out = sum(scale[-1] for scale in mlps[0])
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=npoints[1],
            radius_list=radii[1],
            nsample_list=nsamples[1],
            in_channel=sa1_out,
            mlp_list=mlps[1],
        )
        sa2_out = sum(scale[-1] for scale in mlps[1])
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=sa2_out + 3,
            mlp=global_mlp,
            group_all=True,
        )
        return global_mlp[-1]

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        if point_cloud.ndim != 3:
            raise ValueError("point_cloud must have shape (B, N, C) or (B, C, N)")
        if point_cloud.shape[-1] == self.input_channels:
            point_cloud = point_cloud.permute(0, 2, 1)
        elif point_cloud.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected input_channels={self.input_channels}, got shape {tuple(point_cloud.shape)}"
            )

        xyz = point_cloud[:, :3, :]
        normals = point_cloud[:, 3:6, :] if self.use_normals else None
        l1_xyz, l1_points = self.sa1(xyz, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points.squeeze(-1)


class TwoBranchEarCropRegressor(nn.Module):
    """Shared PointNet++ encoder for left/right ear crops with a joint regression head."""

    def __init__(
        self,
        num_landmarks: int = 170,
        input_channels: int = 6,
        use_normals: bool = True,
        variant: str = "ssg",
        head_channels: Sequence[int] = (512, 256),
        dropout: float = 0.4,
        ssg_npoints: Sequence[int] = (512, 128),
        ssg_radii: Sequence[float] = (0.2, 0.4),
        ssg_nsamples: Sequence[int] = (32, 64),
        ssg_mlps: Sequence[Sequence[int]] = (
            (64, 64, 128),
            (128, 128, 256),
            (256, 512, 1024),
        ),
        msg_npoints: Sequence[int] = (512, 128),
        msg_radii: Sequence[Sequence[float]] = (
            (0.1, 0.2, 0.4),
            (0.2, 0.4, 0.8),
        ),
        msg_nsamples: Sequence[Sequence[int]] = (
            (16, 32, 128),
            (32, 64, 128),
        ),
        msg_mlps: Sequence[Sequence[Sequence[int]]] = (
            ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
            ((64, 64, 128), (128, 128, 256), (128, 128, 256)),
        ),
        msg_global_mlp: Sequence[int] = (256, 512, 1024),
    ):
        super().__init__()
        self.num_landmarks = int(num_landmarks)
        self.encoder = PointNet2FeatureEncoder(
            input_channels=input_channels,
            use_normals=use_normals,
            variant=variant,
            ssg_npoints=ssg_npoints,
            ssg_radii=ssg_radii,
            ssg_nsamples=ssg_nsamples,
            ssg_mlps=ssg_mlps,
            msg_npoints=msg_npoints,
            msg_radii=msg_radii,
            msg_nsamples=msg_nsamples,
            msg_mlps=msg_mlps,
            msg_global_mlp=msg_global_mlp,
        )
        self.regression_head = PointNet2LandmarkRegressor._build_regression_head(
            input_dim=self.encoder.feature_dim * 2,
            hidden_dims=_as_list(head_channels),
            output_dim=self.num_landmarks * 3,
            dropout=float(dropout),
        )

    def forward(self, left_points: torch.Tensor, right_points: torch.Tensor) -> torch.Tensor:
        left_features = self.encoder(left_points)
        right_features = self.encoder(right_points)
        features = torch.cat([left_features, right_features], dim=1)
        landmarks = self.regression_head(features)
        return landmarks.view(left_points.shape[0], self.num_landmarks, 3)


class PointNet2BoxRegressor(nn.Module):
    """Predict tight ear crop box [cx, cy, cz, sx, sy, sz]."""

    def __init__(
        self,
        input_channels: int = 6,
        use_normals: bool = True,
        variant: str = "ssg",
        head_channels: Sequence[int] = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = PointNet2FeatureEncoder(
            input_channels=input_channels,
            use_normals=use_normals,
            variant=variant,
        )
        # Reusing the existing head builder, but change output_dim to 6
        self.regression_head = PointNet2LandmarkRegressor._build_regression_head(
            input_dim=self.encoder.feature_dim,
            hidden_dims=_as_list(head_channels),
            output_dim=6,
            dropout=float(dropout),
        )

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        features = self.encoder(point_cloud)
        box = self.regression_head(features)

        # Force size to be positive using softplus, adding a small epsilon to avoid exactly 0
        center = box[:, :3]
        size = torch.nn.functional.softplus(box[:, 3:]) + 1e-4
        
        return torch.cat([center, size], dim=1)
        
        
def default_model_config() -> dict:
    """Serializable default config for checkpoints and scripts."""
    return {
        "num_landmarks": 170,
        "input_channels": 6,
        "use_normals": True,
        "variant": "ssg",
        "head_channels": [512, 256],
        "dropout": 0.4,
        "ssg_npoints": [512, 128],
        "ssg_radii": [0.2, 0.4],
        "ssg_nsamples": [32, 64],
        "ssg_mlps": [[64, 64, 128], [128, 128, 256], [256, 512, 1024]],
        "msg_npoints": [512, 128],
        "msg_radii": [[0.1, 0.2, 0.4], [0.2, 0.4, 0.8]],
        "msg_nsamples": [[16, 32, 128], [32, 64, 128]],
        "msg_mlps": [
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        ],
        "msg_global_mlp": [256, 512, 1024],
    }
