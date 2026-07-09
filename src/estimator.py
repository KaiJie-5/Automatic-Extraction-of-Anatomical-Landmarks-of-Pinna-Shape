from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from trimesh import Trimesh

from .ear_crop import crop_config_from_dict, sample_crop_point_features
from .pointnet2_model import (
    PointNet2LandmarkRegressor,
    default_model_config,
)
from .preprocessing import (
    compute_mesh_normalization,
    normalize_point_features,
    sample_mesh_surface,
    split_landmark_prediction,
)


class LandmarkExtractor:
    """Landmark extractor implementation."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        num_points: int = 16384,
        seed: int = 0,
        device: str = "cpu",
    ):
        """This function needs to have default values for all arguments. These will be used when instantiating the class
        during the evaluation of your submission."""
        self.checkpoint_path = Path(checkpoint_path)
        self.num_points = int(num_points)
        self.seed = int(seed)
        self.device = torch.device(device)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing trained checkpoint: {self.checkpoint_path}. "
                "Train the model with train_pointnet2.py and write the best checkpoint "
                "to this path before running evaluation."
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_config = default_model_config()
        if isinstance(checkpoint, dict) and "model_config" in checkpoint:
            model_config.update(checkpoint["model_config"])
        self.input_mode = checkpoint.get("input_mode", "full") if isinstance(checkpoint, dict) else "full"
        if isinstance(checkpoint, dict) and "num_points" in checkpoint:
            self.num_points = int(checkpoint["num_points"])
        self.ear_points = int(checkpoint.get("ear_points", 8192)) if isinstance(checkpoint, dict) else 8192
        self.crop_oversample_factor = (
            int(checkpoint.get("crop_oversample_factor", 8)) if isinstance(checkpoint, dict) else 8
        )
        self.crop_max_resample_attempts = (
            int(checkpoint.get("crop_max_resample_attempts", 5))
            if isinstance(checkpoint, dict)
            else 5
        )
        self.crop_min_inside_ratio = (
            float(checkpoint.get("crop_min_inside_ratio", 0.0))
            if isinstance(checkpoint, dict)
            else 0.0
        )
        self.mirror_right_ear = (
            bool(checkpoint.get("mirror_right_ear", False)) if isinstance(checkpoint, dict) else False
        )
        if isinstance(checkpoint, dict) and "seed" in checkpoint:
            self.seed = int(checkpoint["seed"])

        if self.input_mode == "full":
            self.model = PointNet2LandmarkRegressor(**model_config).to(self.device)
            self.crop_config = None
        elif self.input_mode == "ear_crop":
            if not isinstance(checkpoint, dict) or checkpoint.get("crop_config") is None:
                raise ValueError("Ear-crop checkpoint is missing crop_config")
            if int(model_config.get("num_landmarks", 0)) != 85:
                raise ValueError(
                    "Ear-crop checkpoints must use the single-ear 85-landmark model. "
                    "Old two-branch 170-landmark crop checkpoints are not compatible."
                )
            self.model = PointNet2LandmarkRegressor(**model_config).to(self.device)
            self.crop_config = crop_config_from_dict(checkpoint["crop_config"])
        elif self.input_mode == "precropped":
            # No crop_config stored — pre-cropped meshes are supplied externally at inference time.
            self.model = PointNet2LandmarkRegressor(**model_config).to(self.device)
            self.crop_config = None
        else:
            raise ValueError(f"Unsupported checkpoint input_mode: {self.input_mode}")

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        state_dict = {
            key.replace("module.", "", 1): value for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def extract(self, mesh: Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """Method to extract left and right ear landmarks from a 3D mesh. Both output arrays need to be of size (85, 3),
        and need to contain the 85 landmark coordinates for the left and right ear in the correct order.
        This function will be called during the evaluation on the hidden test dataset.
        """
        transform = compute_mesh_normalization(mesh)
        if self.input_mode == "full":
            point_features = sample_mesh_surface(mesh, num_points=self.num_points, seed=self.seed)
            point_features = normalize_point_features(point_features, transform)
            points = torch.from_numpy(point_features).unsqueeze(0).to(self.device)
            with torch.no_grad():
                normalized_landmarks = self.model(points).squeeze(0).cpu().numpy()
        else:
            left_points = sample_crop_point_features(
                mesh=mesh,
                transform=transform,
                crop_box=self.crop_config["left"],
                num_points=self.ear_points,
                seed=self.seed,
                mirror_y=False,
                oversample_factor=self.crop_oversample_factor,
                max_attempts=self.crop_max_resample_attempts,
                min_inside_ratio=self.crop_min_inside_ratio,
            )
            right_points = sample_crop_point_features(
                mesh=mesh,
                transform=transform,
                crop_box=self.crop_config["right"],
                num_points=self.ear_points,
                seed=self.seed + 1,
                mirror_y=self.mirror_right_ear,
                oversample_factor=self.crop_oversample_factor,
                max_attempts=self.crop_max_resample_attempts,
                min_inside_ratio=self.crop_min_inside_ratio,
            )
            left_tensor = torch.from_numpy(left_points).unsqueeze(0).to(self.device)
            right_tensor = torch.from_numpy(right_points).unsqueeze(0).to(self.device)
            with torch.no_grad():
                left_landmarks = self.model(left_tensor).squeeze(0).cpu().numpy()
                right_landmarks = self.model(right_tensor).squeeze(0).cpu().numpy()

            return (
                transform.denormalize_xyz(left_landmarks).astype(np.float32),
                transform.denormalize_xyz(right_landmarks).astype(np.float32),
            )

        landmarks = transform.denormalize_xyz(normalized_landmarks).astype(np.float32)
        return split_landmark_prediction(landmarks)

    def extract_with_crops(
        self,
        mesh: Trimesh,
        left_crop_mesh: Trimesh,
        right_crop_mesh: Trimesh,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference for a ``precropped`` checkpoint.

        Args:
            mesh: Full-body mesh used only to compute the global normalization transform.
            left_crop_mesh: Pre-cropped left-ear mesh (from the box regressor).
            right_crop_mesh: Pre-cropped right-ear mesh (from the box regressor).

        Returns:
            Tuple of (left_landmarks, right_landmarks), each shape (85, 3) in original mm coordinates.
        """
        if self.input_mode != "precropped":
            raise ValueError(
                f"extract_with_crops is only for 'precropped' checkpoints; got '{self.input_mode}'. "
                "Use extract() instead."
            )
        transform = compute_mesh_normalization(mesh)

        left_points = sample_mesh_surface(left_crop_mesh, num_points=self.ear_points, seed=self.seed)
        left_points = normalize_point_features(left_points, transform)

        right_points = sample_mesh_surface(right_crop_mesh, num_points=self.ear_points, seed=self.seed + 1)
        right_points = normalize_point_features(right_points, transform)
        if self.mirror_right_ear:
            right_points[:, 1] *= -1.0
            if right_points.shape[1] >= 5:
                right_points[:, 4] *= -1.0

        left_tensor = torch.from_numpy(left_points).unsqueeze(0).to(self.device)
        right_tensor = torch.from_numpy(right_points).unsqueeze(0).to(self.device)
        with torch.no_grad():
            left_landmarks = self.model(left_tensor).squeeze(0).cpu().numpy()
            right_landmarks = self.model(right_tensor).squeeze(0).cpu().numpy()

        return (
            transform.denormalize_xyz(left_landmarks).astype(np.float32),
            transform.denormalize_xyz(right_landmarks).astype(np.float32),
        )