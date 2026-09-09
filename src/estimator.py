from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from trimesh import Trimesh

from .ear_crop import crop_config_from_dict, sample_crop_point_features
from .calibration import boxes_for_prediction
from .canonical import LocalEarTransform, WorldCropBox, decanonicalize_xyz
from .geometry import sample_canonical_crop
from .pointnet2_model import (
    PointNet2LandmarkRegressor,
    default_model_config,
)
from .precision import checkpoint_autocast_context
from .pointtransformerv3_model import validate_pointtransformerv3_checkpoint_config
from .proposal_models import build_landmark_model, build_locator
from .meshnet import MeshNetLandmarkRegressor, meshnet_inputs
from .shape_prior import (
    BilateralMeanAsymmetryPCAPrior,
    PCAShapePrior,
    gate_bilateral_contours,
    normalise_contour_gate,
)
from .curve import contour_anchor_manifest, decode_connected_curve_paths
from .surface import project_points_to_mesh
from .surface_geometry_features import (
    append_surface_geometry_features,
    surface_geometry_from_model_config,
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
        checkpoint_path: str = "checkpoints/final_pipeline.pt",
        num_points: int = 16384,
        seed: int = 42,
        device: str = "auto",
    ):
        """This function needs to have default values for all arguments. These will be used when instantiating the class
        during the evaluation of your submission."""
        self.checkpoint_path = Path(checkpoint_path)
        self.num_points = int(num_points)
        self.seed = int(seed)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        self.pca_shape_prior = None
        self.bilateral_pca_shape_prior = None
        self.bilateral_pca_contour_gate = ()
        self.curve_path_decoder = None
        self.surface_geometry_config = None

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing trained checkpoint: {self.checkpoint_path}. "
                "Train/package the model with train_pipeline.py and write the final checkpoint "
                "to this path before running evaluation."
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and checkpoint.get("schema_version") == 2:
            self._load_v2(checkpoint)
            return
        self.schema_version = 1
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

    @staticmethod
    def _clean_state_dict(state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    @staticmethod
    def _load_pca_shape_prior(postprocess):
        config = postprocess.get("pca_shape_prior")
        if not isinstance(config, dict) or not bool(config.get("enabled", False)):
            return None
        prior = config.get("prior")
        if not isinstance(prior, dict):
            raise ValueError("pca_shape_prior.prior must be embedded in the checkpoint")
        return PCAShapePrior(
            mean_shape=prior["mean_shape"],
            components=prior["components"],
            n_components=int(prior["n_components"]),
            beta=float(config.get("beta", prior.get("beta", 1.0))),
            landmark_count=int(prior.get("landmark_count", 85)),
            coordinate_frame=str(prior.get("coordinate_frame", "")),
            normalization=str(prior.get("normalization", "")),
        )

    def _apply_pca_shape_prior(self, local_prediction: np.ndarray) -> np.ndarray:
        if self.pca_shape_prior is None:
            return local_prediction
        refined = self.pca_shape_prior.blend(local_prediction)
        if refined.shape != (85, 3) or not np.isfinite(refined).all():
            raise RuntimeError("PCA shape prior returned an invalid (85, 3) prediction")
        return refined

    @staticmethod
    def _load_bilateral_pca_shape_prior(postprocess):
        config = postprocess.get("bilateral_mean_asymmetry_pca_prior")
        if not isinstance(config, dict) or not bool(config.get("enabled", False)):
            return None
        if config.get("ear_order") != ["left", "mirrored_right"]:
            raise ValueError(
                "bilateral PCA ear_order must be ['left', 'mirrored_right']"
            )
        prior = config.get("prior")
        if not isinstance(prior, dict):
            raise ValueError(
                "bilateral_mean_asymmetry_pca_prior.prior must be embedded "
                "in the checkpoint"
            )
        return BilateralMeanAsymmetryPCAPrior(
            common_mean=prior["common_mean"],
            common_components=prior["common_components"],
            common_n_components=int(prior["common_n_components"]),
            asymmetry_mean=prior["asymmetry_mean"],
            asymmetry_components=prior["asymmetry_components"],
            asymmetry_n_components=int(prior["asymmetry_n_components"]),
            common_beta=float(config["common_beta"]),
            asymmetry_beta=float(config["asymmetry_beta"]),
            landmark_count=int(prior.get("landmark_count", 85)),
            coordinate_frame=str(prior.get("coordinate_frame", "")),
            normalization=str(prior.get("normalization", "")),
        )

    @staticmethod
    def _load_bilateral_pca_contour_gate(postprocess):
        config = postprocess.get("bilateral_mean_asymmetry_pca_prior")
        if not isinstance(config, dict) or not bool(config.get("enabled", False)):
            return ()
        return normalise_contour_gate(config.get("contour_gate"))

    def _apply_bilateral_pca_shape_prior(
        self, local_predictions: np.ndarray
    ) -> np.ndarray:
        if self.bilateral_pca_shape_prior is None:
            return local_predictions
        refined = self.bilateral_pca_shape_prior.blend_pair(local_predictions)
        if refined.shape != (2, 85, 3) or not np.isfinite(refined).all():
            raise RuntimeError(
                "bilateral PCA prior returned an invalid (2, 85, 3) prediction"
            )
        return refined

    def _apply_pca_postprocess_pair(
        self, local_predictions: np.ndarray
    ) -> np.ndarray:
        values = np.asarray(local_predictions, dtype=np.float32)
        if values.shape != (2, 85, 3) or not np.isfinite(values).all():
            raise RuntimeError(
                "paired PCA post-processing needs a finite (2, 85, 3) input"
            )
        if self.bilateral_pca_shape_prior is not None:
            bilateral = self._apply_bilateral_pca_shape_prior(values)
            if self.bilateral_pca_contour_gate:
                if self.pca_shape_prior is None:
                    raise RuntimeError(
                        "bilateral PCA contour gating requires an independent "
                        "PCA prior"
                    )
                independent = np.stack(
                    [self._apply_pca_shape_prior(values[index]) for index in range(2)],
                    axis=0,
                )
                return gate_bilateral_contours(
                    independent,
                    bilateral,
                    self.bilateral_pca_contour_gate,
                )
            return bilateral
        if self.pca_shape_prior is not None:
            return np.stack(
                [self._apply_pca_shape_prior(values[index]) for index in range(2)],
                axis=0,
            )
        return values

    def _load_v2(self, checkpoint: dict) -> None:
        required = {
            "locator",
            "landmark",
            "broad_config",
            "crop_calibration",
            "coordinates",
            "sampling",
            "postprocess",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"Incomplete v2 pipeline checkpoint; missing: {missing}")
        for component in ("locator", "landmark"):
            component_missing = sorted(
                {"model_config", "state_dict"} - set(checkpoint[component])
            )
            if component_missing:
                raise ValueError(
                    f"Incomplete v2 {component} component; missing: {component_missing}"
                )
        coordinates = checkpoint["coordinates"]
        if coordinates.get("right_reflection") != [1.0, -1.0, 1.0]:
            raise ValueError("Unsupported v2 right-ear coordinate reflection")
        if coordinates.get("units") != "millimetres":
            raise ValueError("v2 checkpoints must use millimetres")
        self.schema_version = 2
        self.input_mode = "proposal_v2"
        self.locator = build_locator(checkpoint["locator"]["model_config"]).to(self.device)
        landmark_config = dict(checkpoint["landmark"]["model_config"])
        self.landmark_model_config = dict(landmark_config)
        self.surface_geometry_config = surface_geometry_from_model_config(
            self.landmark_model_config
        )
        self.landmark_backbone = landmark_config.get("backbone", "pointnet2")
        self.bilateral_mode = str(
            landmark_config.get("bilateral_mode", "none")
        )
        if self.landmark_backbone == "pointtransformerv3":
            validate_pointtransformerv3_checkpoint_config(
                self.landmark_model_config
            )
        if self.landmark_backbone == "meshnet":
            self.meshnet_target_faces = int(landmark_config.pop("target_faces"))
            landmark_config.pop("backbone")
            self.landmark_model = MeshNetLandmarkRegressor(**landmark_config).to(self.device)
        else:
            self.landmark_model = build_landmark_model(landmark_config).to(self.device)
        self.locator.load_state_dict(
            self._clean_state_dict(checkpoint["locator"]["state_dict"])
        )
        self.landmark_model.load_state_dict(
            self._clean_state_dict(checkpoint["landmark"]["state_dict"])
        )
        self.locator.eval()
        self.landmark_model.eval()
        self.broad_config = checkpoint["broad_config"]
        self.broad_box = WorldCropBox.from_dict(self.broad_config["box"])
        self.initial_center = np.asarray(
            self.broad_config["initial_center"], dtype=np.float32
        )
        self.broad_scale = float(self.broad_config["input_scale"])
        self.crop_calibration = checkpoint["crop_calibration"]
        self.local_scale = float(self.crop_calibration["local_scale"])
        self.locator_points = int(checkpoint["sampling"].get("locator_points", 16384))
        self.landmark_points = int(
            checkpoint["sampling"].get("landmark_points", 16384)
        )
        if "seed" in checkpoint["sampling"]:
            self.seed = int(checkpoint["sampling"]["seed"])
        postprocess = checkpoint["postprocess"]
        self.project_to_surface = bool(postprocess.get("project_to_surface", False))
        curve_path = postprocess.get("curve_path_decoder", {})
        if bool(curve_path.get("enabled", False)):
            if str(self.landmark_model_config.get("decoder", "")) != "surface_curve":
                raise ValueError(
                    "v2 connected curve paths require a surface-curve model"
                )
            if curve_path.get("routing", "section_anchors") != "section_anchors":
                raise ValueError("unsupported v2 connected curve routing mode")
            saved_anchors = curve_path.get("anchor_indices")
            if (
                saved_anchors is not None
                and saved_anchors != contour_anchor_manifest()
            ):
                raise ValueError("v2 connected curve anchor indices do not match")
            field_strength = float(curve_path.get("field_strength", 4.0))
            backtrack_weight = float(curve_path.get("backtrack_weight", 8.0))
            if (
                not np.isfinite(field_strength)
                or field_strength < 0.0
                or not np.isfinite(backtrack_weight)
                or backtrack_weight < 0.0
            ):
                raise ValueError("v2 connected curve path settings are invalid")
            self.curve_path_decoder = {
                "field_strength": field_strength,
                "backtrack_weight": backtrack_weight,
            }
        self.pca_shape_prior = self._load_pca_shape_prior(postprocess)
        self.bilateral_pca_shape_prior = self._load_bilateral_pca_shape_prior(
            postprocess
        )
        self.bilateral_pca_contour_gate = (
            self._load_bilateral_pca_contour_gate(postprocess)
        )
        if (
            self.pca_shape_prior is not None
            and self.bilateral_pca_shape_prior is not None
            and not self.bilateral_pca_contour_gate
        ):
            raise ValueError(
                "v2 bundle cannot enable independent and bilateral PCA priors "
                "simultaneously unless an explicit bilateral contour gate is set"
            )
        if self.bilateral_pca_contour_gate and self.pca_shape_prior is None:
            raise ValueError(
                "bilateral PCA contour gating requires an enabled independent "
                "PCA prior"
            )

    def _extract_v2_ear(self, mesh: Trimesh, ear: str, ear_offset: int) -> np.ndarray:
        broad_features, _, _ = sample_canonical_crop(
            mesh,
            self.broad_box,
            ear,
            self.locator_points,
            self.seed + ear_offset,
        )
        broad_transform = LocalEarTransform(self.broad_box.center, self.broad_scale)
        locator_input = broad_transform.normalize_features(broad_features)
        locator_tensor = torch.from_numpy(locator_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            correction = self.locator(locator_tensor).squeeze(0).float().cpu().numpy()
        predicted_center = self.initial_center + correction
        primary, backup = boxes_for_prediction(predicted_center, self.crop_calibration)
        local_features, crop_mesh, _ = sample_canonical_crop(
            mesh,
            primary,
            ear,
            self.landmark_points,
            self.seed + 100 + ear_offset,
            fallback_box=backup,
            thresholds=self.crop_calibration.get("fallback_thresholds"),
        )
        local_transform = LocalEarTransform(predicted_center, self.local_scale)
        curve_context = None
        with torch.no_grad(), checkpoint_autocast_context(
            self.device, self.landmark_model_config
        ):
            if self.landmark_backbone == "meshnet":
                face_features, neighbors = meshnet_inputs(
                    crop_mesh, self.meshnet_target_faces, ear, local_transform
                )
                local_prediction = self.landmark_model(
                    torch.from_numpy(face_features).unsqueeze(0).to(self.device),
                    torch.from_numpy(neighbors).unsqueeze(0).to(self.device),
                ).squeeze(0).float().cpu().numpy()
            else:
                model_input = append_surface_geometry_features(
                    local_transform.normalize_features(local_features),
                    local_transform.scale,
                    self.surface_geometry_config,
                )
                input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(self.device)
                if self.curve_path_decoder is None:
                    local_prediction = (
                        self.landmark_model(input_tensor)
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy()
                    )
                else:
                    details = self.landmark_model.forward_with_details(
                        input_tensor
                    )
                    local_prediction = (
                        details["final"].squeeze(0).float().cpu().numpy()
                    )
                    curve_context = {
                        "surface_candidates": details["surface_candidates"]
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy(),
                        "curve_logits": details["curve_logits"]
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy(),
                        "curve_arc_coordinates": details[
                            "curve_arc_coordinates"
                        ]
                        .squeeze(0)
                        .float()
                        .cpu()
                        .numpy(),
                        "curve_landmark_fractions": details[
                            "curve_landmark_fractions"
                        ]
                        .float()
                        .cpu()
                        .numpy(),
                    }
        local_prediction = self._apply_pca_shape_prior(local_prediction)
        if curve_context is not None:
            local_prediction, _ = decode_connected_curve_paths(
                crop_mesh,
                ear,
                local_transform,
                curve_context["surface_candidates"],
                curve_context["curve_logits"],
                curve_context["curve_arc_coordinates"],
                local_prediction,
                curve_context["curve_landmark_fractions"],
                **self.curve_path_decoder,
            )
        canonical_prediction = local_transform.denormalize_xyz(local_prediction)
        prediction = decanonicalize_xyz(canonical_prediction, ear).astype(np.float32)
        if self.project_to_surface:
            prediction = project_points_to_mesh(prediction, crop_mesh)
        if prediction.shape != (85, 3) or not np.isfinite(prediction).all():
            raise RuntimeError(f"v2 {ear} prediction is not a finite (85, 3) array")
        return prediction.astype(np.float32)

    def _extract_v2_bilateral(
        self, mesh: Trimesh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run paired model inference and/or paired statistical post-processing."""
        prepared = []
        curve_contexts = []
        for ear_offset, ear in enumerate(("left", "right")):
            broad_features, _, _ = sample_canonical_crop(
                mesh,
                self.broad_box,
                ear,
                self.locator_points,
                self.seed + ear_offset,
            )
            broad_transform = LocalEarTransform(
                self.broad_box.center, self.broad_scale
            )
            locator_input = broad_transform.normalize_features(
                broad_features
            )
            locator_tensor = (
                torch.from_numpy(locator_input).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                correction = (
                    self.locator(locator_tensor)
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            predicted_center = self.initial_center + correction
            primary, backup = boxes_for_prediction(
                predicted_center, self.crop_calibration
            )
            local_features, crop_mesh, _ = sample_canonical_crop(
                mesh,
                primary,
                ear,
                self.landmark_points,
                self.seed + 100 + ear_offset,
                fallback_box=backup,
                thresholds=self.crop_calibration.get("fallback_thresholds"),
            )
            local_transform = LocalEarTransform(
                predicted_center, self.local_scale
            )
            prepared.append(
                (
                    ear,
                    local_transform,
                    crop_mesh,
                    append_surface_geometry_features(
                        local_transform.normalize_features(local_features),
                        local_transform.scale,
                        self.surface_geometry_config,
                    ),
                )
            )

        if self.bilateral_mode != "none":
            if self.landmark_backbone == "meshnet":
                raise RuntimeError(
                    "bilateral neural inference is unavailable for MeshNet"
                )
            paired_input = np.stack(
                [item[3].astype(np.float32) for item in prepared], axis=0
            )
            input_tensor = (
                torch.from_numpy(paired_input).unsqueeze(0).to(self.device)
            )
            with torch.no_grad(), checkpoint_autocast_context(
                self.device, self.landmark_model_config
            ):
                local_predictions = (
                    self.landmark_model(input_tensor)
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
        else:
            # Preserve the established batch-one numerical path for an
            # independent-ear model; only its post-processing is paired.
            predictions = []
            for ear, transform, crop_mesh, model_input in prepared:
                with torch.no_grad(), checkpoint_autocast_context(
                    self.device, self.landmark_model_config
                ):
                    if self.landmark_backbone == "meshnet":
                        face_features, neighbors = meshnet_inputs(
                            crop_mesh,
                            self.meshnet_target_faces,
                            ear,
                            transform,
                        )
                        prediction = self.landmark_model(
                            torch.from_numpy(face_features)
                            .unsqueeze(0)
                            .to(self.device),
                            torch.from_numpy(neighbors)
                            .unsqueeze(0)
                            .to(self.device),
                        )
                    else:
                        input_tensor = (
                            torch.from_numpy(model_input)
                            .unsqueeze(0)
                            .to(self.device)
                        )
                        if self.curve_path_decoder is None:
                            prediction = self.landmark_model(input_tensor)
                        else:
                            details = self.landmark_model.forward_with_details(
                                input_tensor
                            )
                            prediction = details["final"]
                            curve_contexts.append(
                                {
                                    "surface_candidates": details[
                                        "surface_candidates"
                                    ]
                                    .squeeze(0)
                                    .float()
                                    .cpu()
                                    .numpy(),
                                    "curve_logits": details["curve_logits"]
                                    .squeeze(0)
                                    .float()
                                    .cpu()
                                    .numpy(),
                                    "curve_arc_coordinates": details[
                                        "curve_arc_coordinates"
                                    ]
                                    .squeeze(0)
                                    .float()
                                    .cpu()
                                    .numpy(),
                                    "curve_landmark_fractions": details[
                                        "curve_landmark_fractions"
                                    ]
                                    .float()
                                    .cpu()
                                    .numpy(),
                                }
                            )
                        if self.curve_path_decoder is None:
                            curve_contexts.append(None)
                predictions.append(
                    prediction.squeeze(0).float().cpu().numpy()
                )
            local_predictions = np.stack(predictions, axis=0)
        if local_predictions.shape != (2, 85, 3):
            raise RuntimeError(
                "bilateral landmark model did not return shape (2, 85, 3)"
            )

        local_predictions = self._apply_pca_postprocess_pair(local_predictions)
        if self.curve_path_decoder is not None:
            decoded_predictions = []
            for ear_index, (ear, transform, crop_mesh, _) in enumerate(prepared):
                context = curve_contexts[ear_index]
                decoded, _ = decode_connected_curve_paths(
                    crop_mesh,
                    ear,
                    transform,
                    context["surface_candidates"],
                    context["curve_logits"],
                    context["curve_arc_coordinates"],
                    local_predictions[ear_index],
                    context["curve_landmark_fractions"],
                    **self.curve_path_decoder,
                )
                decoded_predictions.append(decoded)
            local_predictions = np.stack(decoded_predictions, axis=0)

        outputs = []
        for ear_index, (ear, transform, crop_mesh, _) in enumerate(prepared):
            local_prediction = local_predictions[ear_index]
            canonical_prediction = transform.denormalize_xyz(
                local_prediction
            )
            prediction = decanonicalize_xyz(
                canonical_prediction, ear
            ).astype(np.float32)
            if self.project_to_surface:
                prediction = project_points_to_mesh(prediction, crop_mesh)
            if prediction.shape != (85, 3) or not np.isfinite(
                prediction
            ).all():
                raise RuntimeError(
                    f"v2 bilateral {ear} prediction is not a finite "
                    "(85, 3) array"
                )
            outputs.append(prediction.astype(np.float32))
        return outputs[0], outputs[1]

    def extract(self, mesh: Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """Method to extract left and right ear landmarks from a 3D mesh. Both output arrays need to be of size (85, 3),
        and need to contain the 85 landmark coordinates for the left and right ear in the correct order.
        This function will be called during the evaluation on the hidden test dataset.
        """
        if self.schema_version == 2:
            if (
                getattr(self, "bilateral_mode", "none") != "none"
                or self.bilateral_pca_shape_prior is not None
            ):
                return self._extract_v2_bilateral(mesh)
            return self._extract_v2_ear(mesh, "left", 0), self._extract_v2_ear(mesh, "right", 1)

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
