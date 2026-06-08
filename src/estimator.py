from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from trimesh import Trimesh

from .ear_crop import crop_config_from_dict, sample_crop_point_features
from .pointnet2_model import (
    PointNet2LandmarkRegressor,
    TwoBranchEarCropRegressor,
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
        self.mirror_right_ear = (
            bool(checkpoint.get("mirror_right_ear", True)) if isinstance(checkpoint, dict) else True
        )
        if isinstance(checkpoint, dict) and "seed" in checkpoint:
            self.seed = int(checkpoint["seed"])

        if self.input_mode == "full":
            self.model = PointNet2LandmarkRegressor(**model_config).to(self.device)
            self.crop_config = None
        elif self.input_mode == "ear_crop":
            if not isinstance(checkpoint, dict) or checkpoint.get("crop_config") is None:
                raise ValueError("Ear-crop checkpoint is missing crop_config")
            self.model = TwoBranchEarCropRegressor(**model_config).to(self.device)
            self.crop_config = crop_config_from_dict(checkpoint["crop_config"])
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
            )
            right_points = sample_crop_point_features(
                mesh=mesh,
                transform=transform,
                crop_box=self.crop_config["right"],
                num_points=self.ear_points,
                seed=self.seed + 1,
                mirror_y=self.mirror_right_ear,
            )
            left_tensor = torch.from_numpy(left_points).unsqueeze(0).to(self.device)
            right_tensor = torch.from_numpy(right_points).unsqueeze(0).to(self.device)
            with torch.no_grad():
                normalized_landmarks = (
                    self.model(left_tensor, right_tensor).squeeze(0).cpu().numpy()
                )

        landmarks = transform.denormalize_xyz(normalized_landmarks).astype(np.float32)
        return split_landmark_prediction(landmarks)
