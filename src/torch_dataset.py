"""PyTorch dataset helpers for pinna landmark regression."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import trimesh
import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np

from .dataset import Dataset as MeshLandmarkDataset
from .ear_crop import EAR_NAMES, make_crop_target, sample_crop_point_features, make_box_target
from .preprocessing import (
    compute_mesh_normalization,
    make_landmark_target,
    normalize_point_features,
    sample_mesh_surface,
)

EAR_NAMES = ("left", "right")

class PinnaPointCloudDataset(TorchDataset):
    """Return normalized sampled point clouds and normalized landmark targets."""

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        num_points: int = 16384,
        seed: int = 0,
        subject_ids: Optional[Sequence[str]] = None,
    ):
        self.base_dataset = MeshLandmarkDataset(mesh_dir=mesh_dir, landmarks_dir=landmarks_dir)
        self.num_points = int(num_points)
        self.seed = int(seed)

        if subject_ids is None:
            self.indices = list(range(len(self.base_dataset)))
        else:
            wanted = {subject_id for subject_id in subject_ids}
            id_to_index = {
                self.base_dataset.get_identifier(idx): idx for idx in range(len(self.base_dataset))
            }
            missing = sorted(wanted - set(id_to_index))
            if missing:
                raise ValueError(f"Unknown subject ids: {missing}")
            self.indices = [id_to_index[subject_id] for subject_id in subject_ids]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        base_idx = self.indices[idx]
        mesh, landmarks_left, landmarks_right = self.base_dataset[base_idx]
        transform = compute_mesh_normalization(mesh)
        point_features = sample_mesh_surface(
            mesh, num_points=self.num_points, seed=self.seed + base_idx
        )
        point_features = normalize_point_features(point_features, transform)
        target = make_landmark_target(landmarks_left, landmarks_right, transform)

        return {
            "points": torch.from_numpy(point_features),
            "landmarks": torch.from_numpy(target),
            "centroid": torch.from_numpy(transform.centroid),
            "scale": torch.tensor(transform.scale, dtype=torch.float32),
            "identifier": self.base_dataset.get_identifier(base_idx),
        }


class PinnaEarCropDataset(TorchDataset):
    """Return one cropped ear point cloud and its global-normalized landmarks."""

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        crop_config: dict,
        ear_points: int = 8192,
        seed: int = 0,
        subject_ids: Optional[Sequence[str]] = None,
        mirror_right_ear: bool = False,
        crop_oversample_factor: int = 8,
        crop_max_resample_attempts: int = 5,
        crop_min_inside_ratio: float = 0.0,
    ):
        self.base_dataset = MeshLandmarkDataset(mesh_dir=mesh_dir, landmarks_dir=landmarks_dir)
        self.crop_config = crop_config
        self.ear_points = int(ear_points)
        self.seed = int(seed)
        self.mirror_right_ear = bool(mirror_right_ear)
        self.crop_oversample_factor = int(crop_oversample_factor)
        self.crop_max_resample_attempts = int(crop_max_resample_attempts)
        self.crop_min_inside_ratio = float(crop_min_inside_ratio)

        if subject_ids is None:
            self.indices = list(range(len(self.base_dataset)))
        else:
            wanted = {subject_id for subject_id in subject_ids}
            id_to_index = {
                self.base_dataset.get_identifier(idx): idx for idx in range(len(self.base_dataset))
            }
            missing = sorted(wanted - set(id_to_index))
            if missing:
                raise ValueError(f"Unknown subject ids: {missing}")
            self.indices = [id_to_index[subject_id] for subject_id in subject_ids]
        self.samples = [
            (base_idx, ear)
            for base_idx in self.indices
            for ear in EAR_NAMES
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        base_idx, ear = self.samples[idx]
        mesh, landmarks_left, landmarks_right = self.base_dataset[base_idx]
        transform = compute_mesh_normalization(mesh)
        ear_offset = 0 if ear == "left" else 1
        landmarks = landmarks_left if ear == "left" else landmarks_right
        point_features = sample_crop_point_features(
            mesh=mesh,
            transform=transform,
            crop_box=self.crop_config[ear],
            num_points=self.ear_points,
            seed=self.seed + base_idx * 2 + ear_offset,
            mirror_y=ear == "right" and self.mirror_right_ear,
            oversample_factor=self.crop_oversample_factor,
            max_attempts=self.crop_max_resample_attempts,
            min_inside_ratio=self.crop_min_inside_ratio,
        )
        target = make_crop_target(landmarks, transform)

        return {
            "points": torch.from_numpy(point_features),
            "landmarks": torch.from_numpy(target),
            "centroid": torch.from_numpy(transform.centroid),
            "scale": torch.tensor(transform.scale, dtype=torch.float32),
            "identifier": self.base_dataset.get_identifier(base_idx),
            "ear": ear,
        }


class PinnaPrecropDataset(TorchDataset):
    """Train from pre-cropped ear meshes produced by the box regressor.

    Expects files named ``{subject_id}_{ear}_mesh.ply`` inside ``cropped_dir``.
    Points are sampled from the pre-cropped mesh but normalised using the
    full-body mesh transform so that landmark targets remain consistent with
    the ``ear_crop`` mode.
    """

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        cropped_dir: str,
        ear_points: int = 8192,
        seed: int = 0,
        subject_ids: Optional[Sequence[str]] = None,
        mirror_right_ear: bool = False,
    ):
        self.base_dataset = MeshLandmarkDataset(mesh_dir=mesh_dir, landmarks_dir=landmarks_dir)
        self.cropped_dir = Path(cropped_dir)
        self.ear_points = int(ear_points)
        self.seed = int(seed)
        self.mirror_right_ear = bool(mirror_right_ear)

        if subject_ids is None:
            self.indices = list(range(len(self.base_dataset)))
        else:
            id_to_index = {
                self.base_dataset.get_identifier(idx): idx for idx in range(len(self.base_dataset))
            }
            missing = sorted(set(subject_ids) - set(id_to_index))
            if missing:
                raise ValueError(f"Unknown subject ids: {missing}")
            self.indices = [id_to_index[sid] for sid in subject_ids]

        self.samples = [
            (base_idx, ear)
            for base_idx in self.indices
            for ear in EAR_NAMES
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        base_idx, ear = self.samples[idx]
        subject_id = self.base_dataset.get_identifier(base_idx)

        # Full mesh → normalization transform + landmarks
        mesh, landmarks_left, landmarks_right = self.base_dataset[base_idx]
        transform = compute_mesh_normalization(mesh)

        # Pre-cropped mesh → sample points from it
        crop_path = self.cropped_dir / f"{subject_id}_{ear}_mesh.ply"
        crop_mesh = trimesh.load(str(crop_path))

        ear_offset = 0 if ear == "left" else 1
        point_features = sample_mesh_surface(
            crop_mesh,
            num_points=self.ear_points,
            seed=self.seed + base_idx * 2 + ear_offset,
        )
        point_features = normalize_point_features(point_features, transform)

        if ear == "right" and self.mirror_right_ear:
            point_features[:, 1] *= -1.0
            if point_features.shape[1] >= 5:
                point_features[:, 4] *= -1.0

        landmarks = landmarks_left if ear == "left" else landmarks_right
        target = make_crop_target(landmarks, transform)

        return {
            "points": torch.from_numpy(point_features),
            "landmarks": torch.from_numpy(target),
            "centroid": torch.from_numpy(transform.centroid),
            "scale": torch.tensor(transform.scale, dtype=torch.float32),
            "identifier": subject_id,
            "ear": ear,
        }


class PinnaEarBoxDataset(torch.utils.data.Dataset):
    """Return broad ear crop point cloud and target ear box."""

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        broad_crop_config: dict,
        ear_points: int = 8192,
        seed: int = 0,
        subject_ids=None,
        box_margin: float = 0.15,
    ):
        self.base_dataset = MeshLandmarkDataset(mesh_dir=mesh_dir, landmarks_dir=landmarks_dir)
        self.broad_crop_config = broad_crop_config
        self.ear_points = int(ear_points)
        self.seed = int(seed)
        self.box_margin = float(box_margin)

        if subject_ids is None:
            self.indices = list(range(len(self.base_dataset)))
        else:
            wanted = {subject_id for subject_id in subject_ids}
            id_to_index = {
                self.base_dataset.get_identifier(idx): idx
                for idx in range(len(self.base_dataset))
            }
            self.indices = [id_to_index[sid] for sid in subject_ids if sid in id_to_index]

        self.samples = [
            (base_idx, ear)
            for base_idx in self.indices
            for ear in EAR_NAMES
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_idx, ear = self.samples[idx]
        mesh, landmarks_left, landmarks_right = self.base_dataset[base_idx]
        transform = compute_mesh_normalization(mesh)

        landmarks = landmarks_left if ear == "left" else landmarks_right
        ear_offset = 0 if ear == "left" else 1

        # Input: broad current crop
        point_features = sample_crop_point_features(
            mesh=mesh,
            transform=transform,
            crop_box=self.broad_crop_config[ear],
            num_points=self.ear_points,
            seed=self.seed + base_idx * 2 + ear_offset,
            mirror_y=False,
        )

        # Target: tight per-subject box from GT landmarks
        target_box = make_box_target(
            landmarks=landmarks,
            transform=transform,
            margin=self.box_margin,
        )

        return {
            "points": torch.from_numpy(point_features).float(),
            "box": torch.from_numpy(target_box).float(),
            "identifier": self.base_dataset.get_identifier(base_idx),
            "ear": ear,
        }
        

def split_subject_ids(
    dataset: MeshLandmarkDataset, val_ratio: float = 0.2, seed: int = 0
) -> Tuple[List[str], List[str]]:
    """Create a deterministic subject-id train/validation split."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    subject_ids = [dataset.get_identifier(idx) for idx in range(len(dataset))]
    if len(subject_ids) == 0:
        raise ValueError("dataset is empty")

    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(subject_ids), generator=generator).tolist()
    shuffled = [subject_ids[idx] for idx in permutation]
    val_count = int(round(len(shuffled) * val_ratio))
    if val_ratio > 0 and val_count == 0 and len(shuffled) > 1:
        val_count = 1

    val_ids = sorted(shuffled[:val_count])
    train_ids = sorted(shuffled[val_count:])
    if not train_ids:
        raise ValueError("validation split leaves no training subjects")
    return train_ids, val_ids


def load_subject_ids(path: Optional[str]) -> Optional[List[str]]:
    """Load subject ids from a newline-delimited text file."""
    if path is None:
        return None
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]