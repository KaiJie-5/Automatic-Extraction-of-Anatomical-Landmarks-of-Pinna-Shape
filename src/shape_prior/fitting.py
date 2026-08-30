"""Build PCA fitting shapes from labelled training ears."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..canonical import LocalEarTransform, canonicalize_xyz
from ..dataset import Dataset
from ..pipeline_dataset import prediction_key


EAR_NAMES = ("left", "right")


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_center_predictions(
    path: str | Path,
    expected_subject_ids: Sequence[str] | None = None,
) -> Mapping[str, np.ndarray]:
    data = read_json(path)
    if data.get("coordinate_frame") != "canonical_mm":
        raise ValueError(
            "centre predictions must declare coordinate_frame='canonical_mm'"
        )
    raw = data.get("center_predictions")
    if not isinstance(raw, Mapping):
        raise ValueError("centre predictions JSON is missing center_predictions")
    if expected_subject_ids is not None:
        expected = {
            f"{subject_id}:{ear}"
            for subject_id in expected_subject_ids
            for ear in EAR_NAMES
        }
        actual = set(raw)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "centre prediction coverage does not match the dataset: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    result = {}
    for key, value in raw.items():
        center = np.asarray(value, dtype=np.float32)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(
                f"centre prediction {key} is not a finite three-value vector"
            )
        result[str(key)] = center
    return result


def select_training_subjects(dataset: Dataset, folds: Mapping[str, object], outer_fold: str) -> list[str]:
    all_subjects = [dataset.get_identifier(index) for index in range(len(dataset))]
    if outer_fold == "final":
        return all_subjects
    fold_index = int(outer_fold)
    for record in folds.get("outer", []):
        if int(record["fold"]) == fold_index:
            return list(record["train"])
    raise ValueError(f"outer fold {outer_fold!r} not found")


def build_training_shapes(
    dataset: Dataset,
    subject_ids: Sequence[str],
    center_predictions: Mapping[str, Sequence[float]],
    calibration: Mapping[str, object],
) -> np.ndarray:
    """Return crop-local canonical normalized landmark shapes `(N, 85, 3)`."""
    subjects = set(dataset.subject_ids)
    missing_subjects = sorted(set(subject_ids) - subjects)
    if missing_subjects:
        raise ValueError(f"unknown subjects for PCA fitting: {missing_subjects[:5]}")

    landmarks_dir = Path(dataset.landmarks_dir)
    local_scale = float(calibration["local_scale"])
    shapes = []
    for subject_id in subject_ids:
        for ear in EAR_NAMES:
            key = prediction_key(subject_id, ear)
            if key not in center_predictions:
                raise ValueError(f"missing centre prediction for PCA fitting: {key}")
            landmarks = Dataset._load_landmarks(
                landmarks_dir / f"{subject_id}_{ear}_ear_landmarks.csv"
            )
            center = np.asarray(center_predictions[key], dtype=np.float32)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError(f"invalid centre prediction for PCA fitting: {key}")
            transform = LocalEarTransform(center, local_scale)
            shapes.append(transform.normalize_xyz(canonicalize_xyz(landmarks, ear)))
    return np.asarray(shapes, dtype=np.float32)
