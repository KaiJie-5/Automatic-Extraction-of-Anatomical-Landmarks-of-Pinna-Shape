"""Dynamic datasets for the proposal-aligned locator and landmark stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from .calibration import boxes_for_prediction
from .canonical import (
    LocalEarTransform,
    WorldCropBox,
    canonicalize_point_features,
    canonicalize_xyz,
    ear_bbox_center,
)
from .dataset import Dataset as MeshLandmarkDataset
from .geometry import sample_canonical_crop
from .meshnet import meshnet_inputs_with_mesh
from .preprocessing import sample_mesh_surface
from .surface_geometry_features import (
    append_surface_geometry_features,
    validate_surface_geometry_config,
)


EAR_NAMES = ("left", "right")


@dataclass(frozen=True)
class PreparedEarGeometry:
    """Exact deterministic validation geometry shared by datasets and viewers."""

    point_features: np.ndarray
    sampled_canonical_features: np.ndarray
    target: np.ndarray
    canonical_landmarks: np.ndarray
    center: np.ndarray
    transform: LocalEarTransform
    primary_box: WorldCropBox
    backup_box: WorldCropBox
    crop_mesh: object
    crop_stats: Mapping[str, object]
    sample_face_indices: Optional[np.ndarray] = None
    sample_barycentric: Optional[np.ndarray] = None


def prepare_ear_geometry(
    mesh,
    landmarks: np.ndarray,
    ear: str,
    center: Sequence[float],
    calibration: Mapping[str, object],
    num_points: int,
    seed: int,
    include_sampling_metadata: bool = False,
    surface_geometry_config: Optional[Mapping[str, object]] = None,
) -> PreparedEarGeometry:
    """Prepare one proposal ear exactly as landmark validation/inference expects."""
    if ear not in EAR_NAMES:
        raise ValueError(f"unsupported ear: {ear}")
    canonical_center = np.asarray(center, dtype=np.float32)
    if canonical_center.shape != (3,) or not np.isfinite(canonical_center).all():
        raise ValueError("predicted ear centre must be a finite three-value vector")
    primary, backup = boxes_for_prediction(canonical_center, calibration)
    sampled_result = sample_canonical_crop(
        mesh,
        primary,
        ear,
        int(num_points),
        int(seed),
        fallback_box=backup,
        thresholds=calibration.get("fallback_thresholds"),
        return_sampling_metadata=include_sampling_metadata,
    )
    sampled, crop_mesh, stats = sampled_result[:3]
    sampling_metadata = sampled_result[3] if include_sampling_metadata else None
    transform = LocalEarTransform(canonical_center, float(calibration["local_scale"]))
    canonical_landmarks = canonicalize_xyz(landmarks, ear).astype(np.float32)
    point_features = transform.normalize_features(sampled).astype(np.float32)
    point_features = append_surface_geometry_features(
        point_features,
        transform.scale,
        surface_geometry_config,
    )
    target = transform.normalize_xyz(canonical_landmarks).astype(np.float32)
    return PreparedEarGeometry(
        point_features=point_features,
        sampled_canonical_features=sampled.astype(np.float32),
        target=target,
        canonical_landmarks=canonical_landmarks,
        center=canonical_center,
        transform=transform,
        primary_box=primary,
        backup_box=backup,
        crop_mesh=crop_mesh,
        crop_stats=dict(stats),
        sample_face_indices=(
            np.asarray(sampling_metadata["face_indices"], dtype=np.int64)
            if sampling_metadata is not None
            else None
        ),
        sample_barycentric=(
            np.asarray(sampling_metadata["barycentric"], dtype=np.float32)
            if sampling_metadata is not None
            else None
        ),
    )


def prediction_key(subject_id: str, ear: str) -> str:
    return f"{subject_id}:{ear}"


def _selected_indices(dataset: MeshLandmarkDataset, subject_ids: Optional[Sequence[str]]):
    id_to_index = {dataset.get_identifier(index): index for index in range(len(dataset))}
    if subject_ids is None:
        return list(range(len(dataset)))
    missing = sorted(set(subject_ids) - set(id_to_index))
    if missing:
        raise ValueError(f"unknown subject ids: {missing}")
    return [id_to_index[item] for item in subject_ids]


class EpochResampledDataset(TorchDataset):
    def __init__(self, seed: int, dynamic_sampling: bool):
        self.seed = int(seed)
        self.dynamic_sampling = bool(dynamic_sampling)
        self.epoch = 0
        # Training counts effective batches in ears.  Ordinary datasets expose
        # one ear per item; bilateral experiments override this with two.
        self.ears_per_item = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def sample_seed(self, sample_index: int, stream: int = 0) -> int:
        epoch = self.epoch if self.dynamic_sampling else 0
        return int(self.seed + sample_index * 1009 + epoch * 1_000_003 + stream * 10_000_019)


class EarLocatorDataset(EpochResampledDataset):
    """Shared left/mirrored-right broad-crop locator samples."""

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        broad_config: Mapping[str, object],
        subject_ids: Optional[Sequence[str]] = None,
        num_points: int = 16384,
        seed: int = 42,
        dynamic_sampling: bool = True,
    ):
        super().__init__(seed, dynamic_sampling)
        self.base = MeshLandmarkDataset(mesh_dir, landmarks_dir)
        self.indices = _selected_indices(self.base, subject_ids)
        self.samples = [(index, ear) for index in self.indices for ear in EAR_NAMES]
        self.num_points = int(num_points)
        self.broad_box = WorldCropBox.from_dict(broad_config["box"])
        self.initial_center = np.asarray(broad_config["initial_center"], dtype=np.float32)
        self.input_scale = float(broad_config["input_scale"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        base_index, ear = self.samples[item]
        mesh, left, right = self.base[base_index]
        landmarks = left if ear == "left" else right
        features, _, stats = sample_canonical_crop(
            mesh,
            self.broad_box,
            ear,
            self.num_points,
            self.sample_seed(item),
        )
        transform = LocalEarTransform(self.broad_box.center, self.input_scale)
        features = transform.normalize_features(features)
        target_center = ear_bbox_center(canonicalize_xyz(landmarks, ear))
        correction = target_center - self.initial_center
        return {
            "points": torch.from_numpy(features),
            "center_correction": torch.from_numpy(correction.astype(np.float32)),
            "true_center": torch.from_numpy(target_center.astype(np.float32)),
            "identifier": self.base.get_identifier(base_index),
            "ear": ear,
            "face_count": stats["face_count"],
            "surface_area": stats["surface_area"],
        }


class EarLandmarkDataset(EpochResampledDataset):
    """Tight crop-local landmark samples centered on out-of-fold locator predictions."""

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        center_predictions: Mapping[str, Sequence[float]],
        calibration: Mapping[str, object],
        subject_ids: Optional[Sequence[str]] = None,
        num_points: int = 16384,
        dense_surface_points: int = 0,
        seed: int = 42,
        dynamic_sampling: bool = True,
        augment: bool = False,
        geodesic_cache_dir: Optional[str] = None,
        include_curve_targets: bool = False,
        surface_geometry_config: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(seed, dynamic_sampling)
        self.base = MeshLandmarkDataset(mesh_dir, landmarks_dir)
        self.indices = _selected_indices(self.base, subject_ids)
        self.samples = [(index, ear) for index in self.indices for ear in EAR_NAMES]
        self.predictions = {
            key: np.asarray(value, dtype=np.float32) for key, value in center_predictions.items()
        }
        self.calibration = calibration
        self.num_points = int(num_points)
        self.dense_surface_points = int(dense_surface_points)
        self.local_scale = float(calibration["local_scale"])
        self.augment = bool(augment)
        self.geodesic_cache_dir = geodesic_cache_dir
        self.include_curve_targets = bool(include_curve_targets)
        self.surface_geometry_config = validate_surface_geometry_config(
            surface_geometry_config
        )
        if self.augment and self.surface_geometry_config["enabled"]:
            raise ValueError(
                "surface geometry features require augmentation to be disabled"
            )
        if self.include_curve_targets and self.geodesic_cache_dir is None:
            raise ValueError("continuous curve targets require a geodesic cache")
        missing = [
            prediction_key(self.base.get_identifier(index), ear)
            for index, ear in self.samples
            if prediction_key(self.base.get_identifier(index), ear) not in self.predictions
        ]
        if missing:
            raise ValueError(f"missing out-of-fold center predictions: {missing[:5]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        base_index, ear = self.samples[item]
        subject_id = self.base.get_identifier(base_index)
        mesh, left, right = self.base[base_index]
        landmarks = left if ear == "left" else right
        center = self.predictions[prediction_key(subject_id, ear)]
        prepared = prepare_ear_geometry(
            mesh, landmarks, ear, center, self.calibration, self.num_points,
            self.sample_seed(item),
            include_sampling_metadata=self.geodesic_cache_dir is not None,
            surface_geometry_config=self.surface_geometry_config,
        )
        features = prepared.point_features.copy()
        target = prepared.target.copy()
        rng = np.random.default_rng(self.sample_seed(item, stream=1))
        augmentation_scale = 1.0
        if self.augment:
            augmentation_scale = float(rng.uniform(0.9, 1.1))
            features[:, :3] *= augmentation_scale
            target *= augmentation_scale
            jitter = np.clip(rng.normal(0.0, 0.005, features[:, :3].shape), -0.02, 0.02)
            features[:, :3] += jitter.astype(np.float32)
        result = {
            "points": torch.from_numpy(features.astype(np.float32)),
            "landmarks": torch.from_numpy(target.astype(np.float32)),
            "scale": torch.tensor(self.local_scale, dtype=torch.float32),
            "center": torch.from_numpy(center.astype(np.float32)),
            "identifier": subject_id,
            "ear": ear,
            "used_backup": prepared.crop_stats["used_backup"],
        }
        if self.geodesic_cache_dir is not None:
            from .geodesic import (
                cache_path,
                load_geodesic_cache_entry,
                sample_geodesic_distances_from_barycentric,
            )

            cached = load_geodesic_cache_entry(
                cache_path(self.geodesic_cache_dir, subject_id, ear),
                prepared.crop_mesh,
            )
            geodesic = sample_geodesic_distances_from_barycentric(
                prepared.crop_mesh,
                prepared.sample_face_indices,
                prepared.sample_barycentric,
                cached,
            )
            result["geodesic_distances_mm"] = torch.from_numpy(
                (geodesic * augmentation_scale).astype(np.float32)
            )
        if self.include_curve_targets:
            from .curve import landmark_arc_fractions

            result["curve_landmark_fractions"] = torch.from_numpy(
                landmark_arc_fractions(prepared.canonical_landmarks)
            )
        if self.dense_surface_points:
            dense = sample_mesh_surface(
                prepared.crop_mesh,
                num_points=self.dense_surface_points,
                seed=self.sample_seed(item, stream=2),
            )
            dense = canonicalize_point_features(dense, ear)
            dense_local = prepared.transform.normalize_xyz(dense[:, :3]) * augmentation_scale
            result["dense_surface"] = torch.from_numpy(dense_local.astype(np.float32))
        return result


class BilateralEarLandmarkDataset(EpochResampledDataset):
    """Subject-paired landmark samples for bilateral fusion experiments.

    Each item contains the canonicalized left and mirrored-right ear in the
    stable order ``(left, right)``.  Ear-specific seeds deliberately reproduce
    the exact seeds used by :class:`EarLandmarkDataset`, so changing to paired
    training does not silently change crop sampling or augmentation.
    """

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        center_predictions: Mapping[str, Sequence[float]],
        calibration: Mapping[str, object],
        subject_ids: Optional[Sequence[str]] = None,
        num_points: int = 16384,
        dense_surface_points: int = 0,
        seed: int = 42,
        dynamic_sampling: bool = True,
        augment: bool = False,
        geodesic_cache_dir: Optional[str] = None,
        include_curve_targets: bool = False,
        surface_geometry_config: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(seed, dynamic_sampling)
        self.ears_per_item = 2
        self.base = MeshLandmarkDataset(mesh_dir, landmarks_dir)
        self.indices = _selected_indices(self.base, subject_ids)
        self.predictions = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in center_predictions.items()
        }
        self.calibration = calibration
        self.num_points = int(num_points)
        self.dense_surface_points = int(dense_surface_points)
        self.local_scale = float(calibration["local_scale"])
        self.augment = bool(augment)
        self.geodesic_cache_dir = geodesic_cache_dir
        self.include_curve_targets = bool(include_curve_targets)
        self.surface_geometry_config = validate_surface_geometry_config(
            surface_geometry_config
        )
        if self.augment and self.surface_geometry_config["enabled"]:
            raise ValueError(
                "surface geometry features require augmentation to be disabled"
            )
        if self.include_curve_targets and self.geodesic_cache_dir is None:
            raise ValueError("continuous curve targets require a geodesic cache")
        missing = [
            prediction_key(self.base.get_identifier(index), ear)
            for index in self.indices
            for ear in EAR_NAMES
            if prediction_key(self.base.get_identifier(index), ear)
            not in self.predictions
        ]
        if missing:
            raise ValueError(
                f"missing out-of-fold center predictions: {missing[:5]}"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        base_index = self.indices[item]
        subject_id = self.base.get_identifier(base_index)
        mesh, left, right = self.base[base_index]
        landmark_sets = (left, right)
        point_features = []
        targets = []
        centers = []
        backups = []
        dense_surfaces = []
        geodesic_surfaces = []
        curve_fractions = []
        for ear_offset, (ear, landmarks) in enumerate(
            zip(EAR_NAMES, landmark_sets)
        ):
            # This is the same flattened item index used by EarLandmarkDataset.
            ear_item = item * len(EAR_NAMES) + ear_offset
            center = self.predictions[prediction_key(subject_id, ear)]
            prepared = prepare_ear_geometry(
                mesh,
                landmarks,
                ear,
                center,
                self.calibration,
                self.num_points,
                self.sample_seed(ear_item),
                include_sampling_metadata=self.geodesic_cache_dir is not None,
                surface_geometry_config=self.surface_geometry_config,
            )
            features = prepared.point_features.copy()
            target = prepared.target.copy()
            augmentation_scale = 1.0
            if self.augment:
                rng = np.random.default_rng(
                    self.sample_seed(ear_item, stream=1)
                )
                augmentation_scale = float(rng.uniform(0.9, 1.1))
                features[:, :3] *= augmentation_scale
                target *= augmentation_scale
                jitter = np.clip(
                    rng.normal(0.0, 0.005, features[:, :3].shape),
                    -0.02,
                    0.02,
                )
                features[:, :3] += jitter.astype(np.float32)
            point_features.append(features.astype(np.float32))
            targets.append(target.astype(np.float32))
            centers.append(center.astype(np.float32))
            backups.append(bool(prepared.crop_stats["used_backup"]))
            if self.geodesic_cache_dir is not None:
                from .geodesic import (
                    cache_path,
                    load_geodesic_cache_entry,
                    sample_geodesic_distances_from_barycentric,
                )

                cached = load_geodesic_cache_entry(
                    cache_path(self.geodesic_cache_dir, subject_id, ear),
                    prepared.crop_mesh,
                )
                geodesic_surfaces.append(
                    sample_geodesic_distances_from_barycentric(
                        prepared.crop_mesh,
                        prepared.sample_face_indices,
                        prepared.sample_barycentric,
                        cached,
                    ).astype(np.float32)
                    * augmentation_scale
                )
            if self.include_curve_targets:
                from .curve import landmark_arc_fractions

                curve_fractions.append(
                    landmark_arc_fractions(prepared.canonical_landmarks)
                )
            if self.dense_surface_points:
                dense = sample_mesh_surface(
                    prepared.crop_mesh,
                    num_points=self.dense_surface_points,
                    seed=self.sample_seed(ear_item, stream=2),
                )
                dense = canonicalize_point_features(dense, ear)
                dense_local = (
                    prepared.transform.normalize_xyz(dense[:, :3])
                    * augmentation_scale
                )
                dense_surfaces.append(dense_local.astype(np.float32))

        result = {
            "points": torch.from_numpy(np.stack(point_features, axis=0)),
            "landmarks": torch.from_numpy(np.stack(targets, axis=0)),
            "scale": torch.full(
                (len(EAR_NAMES),), self.local_scale, dtype=torch.float32
            ),
            "center": torch.from_numpy(np.stack(centers, axis=0)),
            "identifier": subject_id,
            "ear": EAR_NAMES,
            "used_backup": torch.as_tensor(backups, dtype=torch.bool),
        }
        if self.dense_surface_points:
            result["dense_surface"] = torch.from_numpy(
                np.stack(dense_surfaces, axis=0)
            )
        if self.geodesic_cache_dir is not None:
            result["geodesic_distances_mm"] = torch.from_numpy(
                np.stack(geodesic_surfaces, axis=0).astype(np.float32)
            )
        if self.include_curve_targets:
            result["curve_landmark_fractions"] = torch.from_numpy(
                np.stack(curve_fractions, axis=0).astype(np.float32)
            )
        return result


class EarMeshLandmarkDataset(EarLandmarkDataset):
    """Fixed-connectivity MeshNet samples, enabled only after the all-crop gate."""

    def __init__(self, *args, target_faces: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_faces = int(target_faces)
        if self.surface_geometry_config["enabled"]:
            raise ValueError(
                "sampled surface geometry features are unavailable for MeshNet"
            )
        if self.augment:
            raise ValueError("MeshNet screening uses fixed standardized geometry without point jitter")

    def __getitem__(self, item):
        base_index, ear = self.samples[item]
        subject_id = self.base.get_identifier(base_index)
        mesh, left, right = self.base[base_index]
        landmarks = left if ear == "left" else right
        center = self.predictions[prediction_key(subject_id, ear)]
        prepared = prepare_ear_geometry(
            mesh, landmarks, ear, center, self.calibration, 1,
            self.sample_seed(item),
        )
        face_features, neighbors, _ = meshnet_inputs_with_mesh(
            prepared.crop_mesh, self.target_faces, ear, prepared.transform
        )
        result = {
            "face_features": torch.from_numpy(face_features),
            "neighbors": torch.from_numpy(neighbors),
            "landmarks": torch.from_numpy(prepared.target.astype(np.float32)),
            "scale": torch.tensor(self.local_scale, dtype=torch.float32),
            "center": torch.from_numpy(center.astype(np.float32)),
            "identifier": subject_id,
            "ear": ear,
            "used_backup": prepared.crop_stats["used_backup"],
        }
        if self.dense_surface_points:
            dense = sample_mesh_surface(
                prepared.crop_mesh,
                num_points=self.dense_surface_points,
                seed=self.sample_seed(item, stream=2),
            )
            dense = canonicalize_point_features(dense, ear)
            result["dense_surface"] = torch.from_numpy(
                prepared.transform.normalize_xyz(dense[:, :3]).astype(np.float32)
            )
        return result
