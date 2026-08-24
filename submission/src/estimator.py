"""Deterministic inference entry point for the proposal-aligned v2 pipeline."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from trimesh import Trimesh

from .calibration import boxes_for_prediction
from .canonical import LocalEarTransform, WorldCropBox, decanonicalize_xyz
from .geometry import sample_canonical_crop
from .proposal_models import build_landmark_model, build_locator
from .surface import project_points_to_mesh


_REQUIRED_V2_KEYS = {
    "locator",
    "landmark",
    "broad_config",
    "crop_calibration",
    "coordinates",
    "sampling",
    "postprocess",
    "training",
}


def _load_checkpoint(path: Path) -> object:
    """Load our trusted packaged checkpoint across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        return torch.load(path, map_location="cpu")


def _clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint component state_dict must be a nonempty mapping")
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    requested = Path(checkpoint_path).expanduser()
    if requested.is_absolute() or requested.exists():
        return requested.resolve()
    submission_root = Path(__file__).resolve().parents[1]
    return (submission_root / requested).resolve()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return resolved


class LandmarkExtractor:
    """Extract ordered left- and right-ear landmarks from one aligned head mesh."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/final_pipeline.pt",
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        """Load the deterministic final model.

        All parameters have defaults because the challenge evaluator constructs this
        class without arguments. Relative checkpoint paths are resolved from either
        the current directory or the root of the extracted submission.
        """
        self.checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
        self.seed = int(seed)
        self.device = _resolve_device(device)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "Missing final v2 checkpoint at "
                f"{self.checkpoint_path}. Place it at checkpoints/final_pipeline.pt."
            )

        checkpoint = _load_checkpoint(self.checkpoint_path)
        self._load_v2(checkpoint)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def _require_mapping(value: object, name: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError(f"v2 checkpoint field '{name}' must be a mapping")
        return value

    def _load_v2(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, Mapping) or checkpoint.get("schema_version") != 2:
            raise ValueError(
                "This submission accepts only a complete schema-version-2 final "
                "pipeline checkpoint; fold, locator-only, and legacy checkpoints "
                "are unsupported."
            )

        missing = sorted(_REQUIRED_V2_KEYS - set(checkpoint))
        if missing:
            raise ValueError(f"Incomplete v2 pipeline checkpoint; missing: {missing}")

        locator_component = self._require_mapping(checkpoint["locator"], "locator")
        landmark_component = self._require_mapping(checkpoint["landmark"], "landmark")
        for name, component in (
            ("locator", locator_component),
            ("landmark", landmark_component),
        ):
            component_missing = sorted({"model_config", "state_dict"} - set(component))
            if component_missing:
                raise ValueError(
                    f"Incomplete v2 {name} component; missing: {component_missing}"
                )

        coordinates = self._require_mapping(checkpoint["coordinates"], "coordinates")
        if list(coordinates.get("right_reflection", [])) != [1.0, -1.0, 1.0]:
            raise ValueError("Unsupported v2 right-ear coordinate reflection")
        if coordinates.get("units") != "millimetres":
            raise ValueError("The v2 checkpoint must use millimetres")

        training = self._require_mapping(checkpoint["training"], "training")
        subject_ids = training.get("subject_ids")
        if (
            not isinstance(subject_ids, list)
            or len(subject_ids) != 201
            or len(set(subject_ids)) != 201
            or not all(isinstance(value, str) and value for value in subject_ids)
            or int(training.get("subject_count", 0)) != 201
        ):
            raise ValueError("The final v2 checkpoint must contain 201 unique training subjects")
        expected_subject_checksum = hashlib.sha256(
            "\n".join(sorted(subject_ids)).encode("utf-8")
        ).hexdigest()
        if training.get("subject_checksum") != expected_subject_checksum:
            raise ValueError("The v2 training-subject checksum is invalid")

        locator_config = dict(
            self._require_mapping(locator_component["model_config"], "locator.model_config")
        )
        landmark_config = dict(
            self._require_mapping(
                landmark_component["model_config"], "landmark.model_config"
            )
        )
        if landmark_config.get("backbone") != "pointnext":
            raise ValueError(
                "This submission was packaged for the selected PointNeXt final model; "
                f"checkpoint backbone is {landmark_config.get('backbone')!r}."
            )

        self.locator = build_locator(locator_config).to(self.device)
        self.landmark_model = build_landmark_model(landmark_config).to(self.device)
        self.locator.load_state_dict(
            _clean_state_dict(locator_component["state_dict"]), strict=True
        )
        self.landmark_model.load_state_dict(
            _clean_state_dict(landmark_component["state_dict"]), strict=True
        )
        self.locator.eval()
        self.landmark_model.eval()

        broad_config = self._require_mapping(checkpoint["broad_config"], "broad_config")
        self.broad_box = WorldCropBox.from_dict(
            self._require_mapping(broad_config.get("box"), "broad_config.box")
        )
        self.initial_center = np.asarray(
            broad_config.get("initial_center"), dtype=np.float32
        )
        self.broad_scale = float(broad_config.get("input_scale", 0.0))
        if self.initial_center.shape != (3,) or not np.isfinite(self.initial_center).all():
            raise ValueError("broad_config.initial_center must be a finite three-vector")
        if not np.isfinite(self.broad_scale) or self.broad_scale <= 0.0:
            raise ValueError("broad_config.input_scale must be finite and positive")

        self.crop_calibration = dict(
            self._require_mapping(checkpoint["crop_calibration"], "crop_calibration")
        )
        if self.crop_calibration.get("schema_version") != 1:
            raise ValueError("Unsupported crop-calibration schema")
        primary_calibration = self._require_mapping(
            self.crop_calibration.get("primary"), "crop_calibration.primary"
        )
        backup_calibration = self._require_mapping(
            self.crop_calibration.get("backup"), "crop_calibration.backup"
        )
        if float(primary_calibration.get("complete_ear_coverage", 0.0)) < 0.99:
            raise ValueError("The primary crop calibration does not reach 99% coverage")
        if float(backup_calibration.get("complete_ear_coverage", 0.0)) < 1.0:
            raise ValueError("The backup crop calibration does not reach 100% coverage")
        self.local_scale = float(self.crop_calibration.get("local_scale", 0.0))
        coordinate_scale = float(coordinates.get("local_scale", self.local_scale))
        if not np.isfinite(self.local_scale) or self.local_scale <= 0.0:
            raise ValueError("crop_calibration.local_scale must be finite and positive")
        if not np.isclose(coordinate_scale, self.local_scale, rtol=0.0, atol=1e-6):
            raise ValueError(
                "coordinates.local_scale disagrees with crop_calibration.local_scale"
            )
        # Validate both serialized boxes immediately rather than during hidden testing.
        boxes_for_prediction(self.initial_center, self.crop_calibration)

        sampling = self._require_mapping(checkpoint["sampling"], "sampling")
        self.locator_points = int(sampling.get("locator_points", 0))
        self.landmark_points = int(sampling.get("landmark_points", 0))
        if self.locator_points != 16384 or self.landmark_points != 16384:
            raise ValueError("The selected final pipeline requires 16,384 points per stage")
        if "seed" in sampling:
            self.seed = int(sampling["seed"])

        postprocess = self._require_mapping(checkpoint["postprocess"], "postprocess")
        self.project_to_surface = bool(postprocess.get("project_to_surface", False))
        if not self.project_to_surface:
            raise ValueError("The selected final pipeline requires surface projection")

    def _extract_ear(self, mesh: Trimesh, ear: str, ear_offset: int) -> np.ndarray:
        broad_features, _, _ = sample_canonical_crop(
            mesh=mesh,
            canonical_box=self.broad_box,
            ear=ear,
            num_points=self.locator_points,
            seed=self.seed + ear_offset,
        )
        broad_transform = LocalEarTransform(self.broad_box.center, self.broad_scale)
        locator_input = broad_transform.normalize_features(broad_features)
        locator_tensor = torch.from_numpy(locator_input).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            correction = (
                self.locator(locator_tensor).squeeze(0).float().cpu().numpy()
            )
        if correction.shape != (3,) or not np.isfinite(correction).all():
            raise RuntimeError(f"The {ear} locator did not return a finite three-vector")

        predicted_center = self.initial_center + correction
        primary_box, backup_box = boxes_for_prediction(
            predicted_center, self.crop_calibration
        )
        local_features, crop_mesh, _ = sample_canonical_crop(
            mesh=mesh,
            canonical_box=primary_box,
            ear=ear,
            num_points=self.landmark_points,
            seed=self.seed + 100 + ear_offset,
            fallback_box=backup_box,
            thresholds=self.crop_calibration.get("fallback_thresholds"),
        )

        local_transform = LocalEarTransform(predicted_center, self.local_scale)
        model_input = local_transform.normalize_features(local_features)
        input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            local_prediction = (
                self.landmark_model(input_tensor).squeeze(0).float().cpu().numpy()
            )

        canonical_prediction = local_transform.denormalize_xyz(local_prediction)
        prediction = decanonicalize_xyz(canonical_prediction, ear).astype(np.float32)
        if self.project_to_surface:
            prediction = project_points_to_mesh(prediction, crop_mesh)
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (85, 3) or not np.isfinite(prediction).all():
            raise RuntimeError(f"The {ear} prediction is not a finite (85, 3) array")
        return prediction

    def extract(self, mesh: Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """Return left and right `(85, 3)` landmarks in official head coordinates."""
        if not isinstance(mesh, Trimesh):
            raise TypeError("mesh must be a trimesh.Trimesh")
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("mesh must contain finite vertices with shape (N, 3)")
        if (
            faces.ndim != 2
            or faces.shape[1:] != (3,)
            or not len(faces)
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
        ):
            raise ValueError("mesh must contain valid triangular faces")

        left = self._extract_ear(mesh, "left", 0)
        right = self._extract_ear(mesh, "right", 1)
        return left, right
