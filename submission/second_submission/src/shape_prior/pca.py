"""Small NumPy PCA prior for crop-local canonical landmark predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


LANDMARK_COUNT = 85
COORDINATE_FRAME = "crop_local_canonical_normalized"
NORMALIZATION = "centroid_rms"
EPSILON = 1e-8


def _normalise_shape(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (LANDMARK_COUNT, 3) or not np.isfinite(values).all():
        raise ValueError("landmarks must be finite with shape (85, 3)")
    centroid = values.mean(axis=0)
    centered = values - centroid
    scale = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    if scale < EPSILON:
        scale = 1.0
    return centered / scale, centroid.astype(np.float32), scale


@dataclass
class PCAShapePrior:
    """PCA projection prior in the landmark model's local output frame."""

    mean_shape: np.ndarray
    components: np.ndarray
    n_components: int
    beta: float = 1.0
    landmark_count: int = LANDMARK_COUNT
    coordinate_frame: str = COORDINATE_FRAME
    normalization: str = NORMALIZATION

    def __post_init__(self) -> None:
        self.landmark_count = int(self.landmark_count)
        if self.landmark_count != LANDMARK_COUNT:
            raise ValueError("only 85-landmark pinna priors are supported")
        self.mean_shape = np.asarray(self.mean_shape, dtype=np.float32).reshape(-1)
        self.components = np.asarray(self.components, dtype=np.float32)
        if self.components.size == 0:
            self.components = self.components.reshape(0, LANDMARK_COUNT * 3)
        if (
            self.mean_shape.shape != (LANDMARK_COUNT * 3,)
            or self.components.ndim != 2
            or self.components.shape[1] != LANDMARK_COUNT * 3
            or not np.isfinite(self.mean_shape).all()
            or not np.isfinite(self.components).all()
        ):
            raise ValueError("invalid PCA shape-prior arrays")
        self.n_components = min(int(self.n_components), len(self.components))
        self.beta = float(self.beta)
        if self.coordinate_frame != COORDINATE_FRAME or self.normalization != NORMALIZATION:
            raise ValueError("unsupported PCA shape-prior coordinate metadata")
        if (
            self.n_components <= 0
            or not np.isfinite(self.beta)
            or not 0.0 <= self.beta <= 1.0
        ):
            raise ValueError("invalid PCA shape-prior settings")

    @classmethod
    def fit(cls, shapes: np.ndarray, n_components: int = 32, beta: float = 1.0) -> "PCAShapePrior":
        """Fit from an array of crop-local canonical shapes with shape `(N, 85, 3)`."""
        values = np.asarray(shapes, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (LANDMARK_COUNT, 3):
            raise ValueError("training shapes must have shape (N, 85, 3)")
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("PCA shape prior needs at least two finite training ears")
        matrix = np.stack([_normalise_shape(shape)[0].reshape(-1) for shape in values])
        mean_shape = matrix.mean(axis=0).astype(np.float32)
        _, _, vt = np.linalg.svd(matrix - mean_shape, full_matrices=False)
        count = min(int(n_components), len(vt))
        if count <= 0:
            raise ValueError("n_components must be positive")
        return cls(mean_shape, vt[:count].astype(np.float32), count, beta)

    def project_one(self, prediction: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Project one `(85, 3)` prediction into PCA space and return `(85, 3)`."""
        normalised, centroid, scale = _normalise_shape(prediction)
        flat = normalised.reshape(-1)
        count = self.n_components if n_components is None else min(int(n_components), len(self.components))
        basis = self.components[:count]
        projected = self.mean_shape + (flat - self.mean_shape) @ basis.T @ basis
        return (projected.reshape(LANDMARK_COUNT, 3) * scale + centroid).astype(np.float32)

    def project_batch(self, predictions: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Project a batch of predictions with shape `(B, 85, 3)`."""
        values = np.asarray(predictions, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (LANDMARK_COUNT, 3):
            raise ValueError("batch predictions must have shape (B, 85, 3)")
        return np.stack([self.project_one(item, n_components) for item in values])

    def blend(self, prediction: np.ndarray, beta: float | None = None, n_components: int | None = None) -> np.ndarray:
        """Blend one prediction or a batch with its PCA projection."""
        values = np.asarray(prediction, dtype=np.float32)
        weight = self.beta if beta is None else float(beta)
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("PCA blend beta must be finite and in [0, 1]")
        projected = (
            self.project_one(values, n_components)
            if values.ndim == 2
            else self.project_batch(values, n_components)
        )
        return ((1.0 - weight) * values + weight * projected).astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save all inference-time prior data to a compressed `.npz` file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            schema_version=np.asarray(1, dtype=np.int32),
            mean_shape=self.mean_shape.astype(np.float32),
            components=self.components.astype(np.float32),
            n_components=np.asarray(self.n_components, dtype=np.int32),
            beta=np.asarray(self.beta, dtype=np.float32),
            landmark_count=np.asarray(self.landmark_count, dtype=np.int32),
            coordinate_frame=np.asarray(self.coordinate_frame),
            normalization=np.asarray(self.normalization),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PCAShapePrior":
        """Load a prior saved by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            if int(np.asarray(data["schema_version"]).item()) != 1:
                raise ValueError("unsupported PCA shape-prior schema")
            return cls(
                mean_shape=data["mean_shape"],
                components=data["components"],
                n_components=int(np.asarray(data["n_components"]).item()),
                beta=float(np.asarray(data["beta"]).item()),
                landmark_count=int(np.asarray(data["landmark_count"]).item()),
                coordinate_frame=str(np.asarray(data["coordinate_frame"]).item()),
                normalization=str(np.asarray(data["normalization"]).item()),
            )
