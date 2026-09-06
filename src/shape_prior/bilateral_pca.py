"""Bilateral PCA in shared-morphology and signed-asymmetry coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pca import COORDINATE_FRAME, LANDMARK_COUNT, _normalise_shape


BILATERAL_NORMALIZATION = "per_ear_centroid_rms_then_mean_asymmetry"
PAIR_SHAPE = (2, LANDMARK_COUNT, 3)
FLAT_DIMENSION = LANDMARK_COUNT * 3


def _fit_basis(matrix: np.ndarray, requested: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != FLAT_DIMENSION:
        raise ValueError("bilateral PCA matrix has an invalid shape")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bilateral PCA needs at least two finite subject pairs")
    count = min(int(requested), len(values))
    if count <= 0:
        raise ValueError("PCA component counts must be positive")
    mean = values.mean(axis=0).astype(np.float32)
    _, _, right_vectors = np.linalg.svd(values - mean, full_matrices=False)
    return mean, right_vectors[:count].astype(np.float32)


def _project_vector(
    value: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    count: int,
) -> np.ndarray:
    basis = components[:count]
    return (mean + (value - mean) @ basis.T @ basis).astype(np.float32)


def _validate_beta(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


def _validate_component_count(
    value: int | None,
    available: int,
    default: int,
    name: str,
) -> int:
    result = default if value is None else int(value)
    if result <= 0 or result > available:
        raise ValueError(f"{name} must be in [1, {available}]")
    return result


def _normalise_pair(
    pair: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(pair, dtype=np.float32)
    if values.shape != PAIR_SHAPE or not np.isfinite(values).all():
        raise ValueError("bilateral landmarks must be finite with shape (2, 85, 3)")
    normalised = []
    centroids = []
    scales = []
    for ear in values:
        shape, centroid, scale = _normalise_shape(ear)
        normalised.append(shape)
        centroids.append(centroid)
        scales.append(scale)
    return (
        np.asarray(normalised, dtype=np.float32),
        np.asarray(centroids, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


@dataclass
class BilateralMeanAsymmetryPCAPrior:
    """Low-rank prior for a canonical ``(left, mirrored-right)`` ear pair.

    Each ear is independently centred and RMS-normalised first.  The two
    aligned shapes are then represented by their common morphology
    ``(left + right) / 2`` and signed asymmetry ``(left - right) / 2``.
    Separate bases and blend strengths keep strong common-shape regularisation
    from erasing genuine left/right differences.
    """

    common_mean: np.ndarray
    common_components: np.ndarray
    common_n_components: int
    asymmetry_mean: np.ndarray
    asymmetry_components: np.ndarray
    asymmetry_n_components: int
    common_beta: float = 1.0
    asymmetry_beta: float = 1.0
    landmark_count: int = LANDMARK_COUNT
    coordinate_frame: str = COORDINATE_FRAME
    normalization: str = BILATERAL_NORMALIZATION

    def __post_init__(self) -> None:
        self.landmark_count = int(self.landmark_count)
        if self.landmark_count != LANDMARK_COUNT:
            raise ValueError("only paired 85-landmark pinna priors are supported")
        self.common_mean = np.asarray(self.common_mean, dtype=np.float32).reshape(-1)
        self.asymmetry_mean = np.asarray(
            self.asymmetry_mean, dtype=np.float32
        ).reshape(-1)
        self.common_components = np.asarray(
            self.common_components, dtype=np.float32
        )
        self.asymmetry_components = np.asarray(
            self.asymmetry_components, dtype=np.float32
        )
        for name, mean, components in (
            ("common", self.common_mean, self.common_components),
            ("asymmetry", self.asymmetry_mean, self.asymmetry_components),
        ):
            if (
                mean.shape != (FLAT_DIMENSION,)
                or components.ndim != 2
                or components.shape[1] != FLAT_DIMENSION
                or not len(components)
                or not np.isfinite(mean).all()
                or not np.isfinite(components).all()
            ):
                raise ValueError(f"invalid bilateral {name} PCA arrays")
        self.common_n_components = _validate_component_count(
            self.common_n_components,
            len(self.common_components),
            len(self.common_components),
            "common_n_components",
        )
        self.asymmetry_n_components = _validate_component_count(
            self.asymmetry_n_components,
            len(self.asymmetry_components),
            len(self.asymmetry_components),
            "asymmetry_n_components",
        )
        self.common_beta = _validate_beta(self.common_beta, "common_beta")
        self.asymmetry_beta = _validate_beta(
            self.asymmetry_beta, "asymmetry_beta"
        )
        if (
            self.coordinate_frame != COORDINATE_FRAME
            or self.normalization != BILATERAL_NORMALIZATION
        ):
            raise ValueError("unsupported bilateral PCA coordinate metadata")

    @classmethod
    def fit(
        cls,
        pairs: np.ndarray,
        common_components: int = 64,
        asymmetry_components: int = 32,
        common_beta: float = 1.0,
        asymmetry_beta: float = 1.0,
    ) -> "BilateralMeanAsymmetryPCAPrior":
        values = np.asarray(pairs, dtype=np.float32)
        if values.ndim != 4 or values.shape[1:] != PAIR_SHAPE:
            raise ValueError("training pairs must have shape (N, 2, 85, 3)")
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("bilateral PCA needs at least two finite subject pairs")
        common_rows = []
        asymmetry_rows = []
        for pair in values:
            normalised, _, _ = _normalise_pair(pair)
            left, right = normalised
            common_rows.append(((left + right) * 0.5).reshape(-1))
            asymmetry_rows.append(((left - right) * 0.5).reshape(-1))
        common_mean, common_basis = _fit_basis(
            np.asarray(common_rows), common_components
        )
        asymmetry_mean, asymmetry_basis = _fit_basis(
            np.asarray(asymmetry_rows), asymmetry_components
        )
        return cls(
            common_mean=common_mean,
            common_components=common_basis,
            common_n_components=len(common_basis),
            asymmetry_mean=asymmetry_mean,
            asymmetry_components=asymmetry_basis,
            asymmetry_n_components=len(asymmetry_basis),
            common_beta=common_beta,
            asymmetry_beta=asymmetry_beta,
        )

    def blend_pair(
        self,
        prediction: np.ndarray,
        common_beta: float | None = None,
        asymmetry_beta: float | None = None,
        common_components: int | None = None,
        asymmetry_components: int | None = None,
    ) -> np.ndarray:
        normalised, centroids, scales = _normalise_pair(prediction)
        common_weight = _validate_beta(
            self.common_beta if common_beta is None else common_beta,
            "common_beta",
        )
        asymmetry_weight = _validate_beta(
            self.asymmetry_beta if asymmetry_beta is None else asymmetry_beta,
            "asymmetry_beta",
        )
        common_count = _validate_component_count(
            common_components,
            len(self.common_components),
            self.common_n_components,
            "common_components",
        )
        asymmetry_count = _validate_component_count(
            asymmetry_components,
            len(self.asymmetry_components),
            self.asymmetry_n_components,
            "asymmetry_components",
        )
        left, right = normalised
        common = ((left + right) * 0.5).reshape(-1)
        asymmetry = ((left - right) * 0.5).reshape(-1)
        projected_common = _project_vector(
            common, self.common_mean, self.common_components, common_count
        )
        projected_asymmetry = _project_vector(
            asymmetry,
            self.asymmetry_mean,
            self.asymmetry_components,
            asymmetry_count,
        )
        common = (
            (1.0 - common_weight) * common
            + common_weight * projected_common
        )
        asymmetry = (
            (1.0 - asymmetry_weight) * asymmetry
            + asymmetry_weight * projected_asymmetry
        )
        common = common.reshape(LANDMARK_COUNT, 3)
        asymmetry = asymmetry.reshape(LANDMARK_COUNT, 3)
        reconstructed = np.stack(
            [common + asymmetry, common - asymmetry], axis=0
        )
        return (
            reconstructed * scales[:, None, None]
            + centroids[:, None, :]
        ).astype(np.float32)

    def blend(
        self,
        predictions: np.ndarray,
        **overrides,
    ) -> np.ndarray:
        """Blend one pair or a batch of pairs while retaining pair boundaries."""
        values = np.asarray(predictions, dtype=np.float32)
        if values.shape == PAIR_SHAPE:
            return self.blend_pair(values, **overrides)
        if values.ndim != 4 or values.shape[1:] != PAIR_SHAPE:
            raise ValueError(
                "bilateral predictions must have shape (2, 85, 3) or "
                "(B, 2, 85, 3)"
            )
        return np.stack([self.blend_pair(pair, **overrides) for pair in values])

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            schema_version=np.asarray(1, dtype=np.int32),
            prior_type=np.asarray("bilateral_mean_asymmetry_pca"),
            common_mean=self.common_mean.astype(np.float32),
            common_components=self.common_components.astype(np.float32),
            common_n_components=np.asarray(
                self.common_n_components, dtype=np.int32
            ),
            asymmetry_mean=self.asymmetry_mean.astype(np.float32),
            asymmetry_components=self.asymmetry_components.astype(np.float32),
            asymmetry_n_components=np.asarray(
                self.asymmetry_n_components, dtype=np.int32
            ),
            common_beta=np.asarray(self.common_beta, dtype=np.float32),
            asymmetry_beta=np.asarray(self.asymmetry_beta, dtype=np.float32),
            landmark_count=np.asarray(self.landmark_count, dtype=np.int32),
            coordinate_frame=np.asarray(self.coordinate_frame),
            normalization=np.asarray(self.normalization),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BilateralMeanAsymmetryPCAPrior":
        with np.load(path, allow_pickle=False) as data:
            if int(np.asarray(data["schema_version"]).item()) != 1:
                raise ValueError("unsupported bilateral PCA schema")
            if str(np.asarray(data["prior_type"]).item()) != (
                "bilateral_mean_asymmetry_pca"
            ):
                raise ValueError("artifact is not a bilateral mean/asymmetry prior")
            return cls(
                common_mean=data["common_mean"],
                common_components=data["common_components"],
                common_n_components=int(
                    np.asarray(data["common_n_components"]).item()
                ),
                asymmetry_mean=data["asymmetry_mean"],
                asymmetry_components=data["asymmetry_components"],
                asymmetry_n_components=int(
                    np.asarray(data["asymmetry_n_components"]).item()
                ),
                common_beta=float(np.asarray(data["common_beta"]).item()),
                asymmetry_beta=float(
                    np.asarray(data["asymmetry_beta"]).item()
                ),
                landmark_count=int(np.asarray(data["landmark_count"]).item()),
                coordinate_frame=str(
                    np.asarray(data["coordinate_frame"]).item()
                ),
                normalization=str(np.asarray(data["normalization"]).item()),
            )
