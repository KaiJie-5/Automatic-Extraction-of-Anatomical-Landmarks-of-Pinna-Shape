"""Coordinate transforms shared by training and v2 inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


RIGHT_REFLECTION = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)


def canonicalize_xyz(xyz: np.ndarray, ear: str) -> np.ndarray:
    values = np.asarray(xyz, dtype=np.float32)
    if values.shape[-1] != 3:
        raise ValueError("xyz must end in three coordinate channels")
    if ear not in {"left", "right"}:
        raise ValueError("ear must be 'left' or 'right'")
    return values.copy() if ear == "left" else values * RIGHT_REFLECTION


def decanonicalize_xyz(xyz: np.ndarray, ear: str) -> np.ndarray:
    return canonicalize_xyz(xyz, ear)


def canonicalize_point_features(features: np.ndarray, ear: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("features must have shape (N, C>=3)")
    values[:, :3] = canonicalize_xyz(values[:, :3], ear)
    if values.shape[1] >= 6:
        values[:, 3:6] = canonicalize_xyz(values[:, 3:6], ear)
    return values


def ear_bbox_center(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (85, 3) or not np.isfinite(points).all():
        raise ValueError("landmarks must be finite with shape (85, 3)")
    return ((points.min(axis=0) + points.max(axis=0)) * 0.5).astype(np.float32)


@dataclass(frozen=True)
class LocalEarTransform:
    center: np.ndarray
    scale: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float32)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite 3-vector")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", float(self.scale))

    def normalize_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return ((np.asarray(xyz, dtype=np.float32) - self.center) / self.scale).astype(np.float32)

    def denormalize_xyz(self, xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float32) * self.scale + self.center).astype(np.float32)

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32).copy()
        values[:, :3] = self.normalize_xyz(values[:, :3])
        return values

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LocalEarTransform":
        return cls(np.asarray(data["center"], dtype=np.float32), float(data["scale"]))


@dataclass(frozen=True)
class WorldCropBox:
    """Axis-aligned box expressed in canonical challenge millimetres."""

    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float32)
        maximum = np.asarray(self.maximum, dtype=np.float32)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("crop bounds must be three-vectors")
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("crop bounds must be finite")
        if np.any(maximum <= minimum):
            raise ValueError("crop maximum must exceed minimum on every axis")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    @property
    def half_extent(self) -> np.ndarray:
        return (self.maximum - self.minimum) * 0.5

    @property
    def scale(self) -> float:
        return float(np.max(self.half_extent))

    def for_ear(self, ear: str) -> "WorldCropBox":
        if ear == "left":
            return self
        if ear != "right":
            raise ValueError("ear must be 'left' or 'right'")
        reflected_min = canonicalize_xyz(self.maximum, ear)
        reflected_max = canonicalize_xyz(self.minimum, ear)
        return WorldCropBox(np.minimum(reflected_min, reflected_max), np.maximum(reflected_min, reflected_max))

    def contains(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32)
        return np.all((values >= self.minimum) & (values <= self.maximum), axis=-1)

    def to_dict(self) -> dict:
        return {"min": self.minimum.tolist(), "max": self.maximum.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[float]]) -> "WorldCropBox":
        return cls(np.asarray(data["min"], dtype=np.float32), np.asarray(data["max"], dtype=np.float32))


def box_from_center(center: np.ndarray, negative: np.ndarray, positive: np.ndarray) -> WorldCropBox:
    center = np.asarray(center, dtype=np.float32)
    return WorldCropBox(center - np.asarray(negative, dtype=np.float32), center + np.asarray(positive, dtype=np.float32))
