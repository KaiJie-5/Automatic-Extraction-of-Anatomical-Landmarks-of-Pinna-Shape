"""Dynamic datasets for the proposal-aligned locator and landmark stages."""

from __future__ import annotations

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
from .meshnet import meshnet_inputs
from .preprocessing import sample_mesh_surface


EAR_NAMES = ("left", "right")


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
        primary, backup = boxes_for_prediction(center, self.calibration)
        features, crop_mesh, stats = sample_canonical_crop(
            mesh,
            primary,
            ear,
            self.num_points,
            self.sample_seed(item),
            fallback_box=backup,
            thresholds=self.calibration.get("fallback_thresholds"),
        )
        transform = LocalEarTransform(center, self.local_scale)
        features = transform.normalize_features(features)
        target = transform.normalize_xyz(canonicalize_xyz(landmarks, ear))
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
            "used_backup": stats["used_backup"],
        }
        if self.dense_surface_points:
            dense = sample_mesh_surface(
                crop_mesh,
                num_points=self.dense_surface_points,
                seed=self.sample_seed(item, stream=2),
            )
            dense = canonicalize_point_features(dense, ear)
            dense_local = transform.normalize_xyz(dense[:, :3]) * augmentation_scale
            result["dense_surface"] = torch.from_numpy(dense_local.astype(np.float32))
        return result


class EarMeshLandmarkDataset(EarLandmarkDataset):
    """Fixed-connectivity MeshNet samples, enabled only after the all-crop gate."""

    def __init__(self, *args, target_faces: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_faces = int(target_faces)
        if self.augment:
            raise ValueError("MeshNet screening uses fixed standardized geometry without point jitter")

    def __getitem__(self, item):
        base_index, ear = self.samples[item]
        subject_id = self.base.get_identifier(base_index)
        mesh, left, right = self.base[base_index]
        landmarks = left if ear == "left" else right
        center = self.predictions[prediction_key(subject_id, ear)]
        primary, backup = boxes_for_prediction(center, self.calibration)
        _, crop_mesh, stats = sample_canonical_crop(
            mesh,
            primary,
            ear,
            num_points=1,
            seed=self.sample_seed(item),
            fallback_box=backup,
            thresholds=self.calibration.get("fallback_thresholds"),
        )
        transform = LocalEarTransform(center, self.local_scale)
        face_features, neighbors = meshnet_inputs(
            crop_mesh, self.target_faces, ear, transform
        )
        target = transform.normalize_xyz(canonicalize_xyz(landmarks, ear))
        result = {
            "face_features": torch.from_numpy(face_features),
            "neighbors": torch.from_numpy(neighbors),
            "landmarks": torch.from_numpy(target.astype(np.float32)),
            "scale": torch.tensor(self.local_scale, dtype=torch.float32),
            "center": torch.from_numpy(center.astype(np.float32)),
            "identifier": subject_id,
            "ear": ear,
            "used_backup": stats["used_backup"],
        }
        if self.dense_surface_points:
            dense = sample_mesh_surface(
                crop_mesh,
                num_points=self.dense_surface_points,
                seed=self.sample_seed(item, stream=2),
            )
            dense = canonicalize_point_features(dense, ear)
            result["dense_surface"] = torch.from_numpy(
                transform.normalize_xyz(dense[:, :3]).astype(np.float32)
            )
        return result
