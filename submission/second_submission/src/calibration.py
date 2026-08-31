"""Leakage-safe broad-region and directional tight-crop calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .canonical import WorldCropBox, box_from_center, canonicalize_xyz, ear_bbox_center


@dataclass(frozen=True)
class EarCalibrationRecord:
    subject_id: str
    ear: str
    landmarks: np.ndarray
    predicted_center: np.ndarray

    def canonical_landmarks(self) -> np.ndarray:
        return canonicalize_xyz(self.landmarks, self.ear)

    def canonical_prediction(self) -> np.ndarray:
        return canonicalize_xyz(self.predicted_center, self.ear)


def fit_broad_box(
    fitting_landmarks: Iterable[np.ndarray],
    calibration_landmarks: Iterable[np.ndarray],
    margins: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
) -> dict:
    fitting = [np.asarray(value, dtype=np.float32) for value in fitting_landmarks]
    heldout = [np.asarray(value, dtype=np.float32) for value in calibration_landmarks]
    if not fitting or not heldout:
        raise ValueError("fitting and calibration ears must both be nonempty")
    points = np.concatenate(fitting, axis=0)
    centers = np.stack([ear_bbox_center(item) for item in fitting])
    center = np.median(centers, axis=0).astype(np.float32)
    base_half = np.maximum(center - points.min(axis=0), points.max(axis=0) - center)
    selected = None
    selected_box = None
    for margin in margins:
        half = base_half * (1.0 + float(margin))
        candidate = WorldCropBox(center - half, center + half)
        if all(bool(candidate.contains(item).all()) for item in heldout):
            selected = float(margin)
            selected_box = candidate
            break
    if selected_box is None:
        selected = float(margins[-1])
        half = base_half * (1.0 + selected)
        selected_box = WorldCropBox(center - half, center + half)
        if not all(bool(selected_box.contains(item).all()) for item in heldout):
            raise ValueError("even the 100% broad-box expansion misses calibration landmarks")
    return {
        "box": selected_box.to_dict(),
        "initial_center": center.tolist(),
        "input_scale": selected_box.scale,
        "margin": selected,
        "calibration_coverage": 1.0,
    }


def fit_broad_box_with_margin(
    fitting_landmarks: Iterable[np.ndarray], margin: float
) -> dict:
    fitting = [np.asarray(value, dtype=np.float32) for value in fitting_landmarks]
    if not fitting:
        raise ValueError("at least one fitting ear is required")
    points = np.concatenate(fitting, axis=0)
    centers = np.stack([ear_bbox_center(item) for item in fitting])
    center = np.median(centers, axis=0).astype(np.float32)
    base_half = np.maximum(center - points.min(axis=0), points.max(axis=0) - center)
    half = base_half * (1.0 + float(margin))
    box = WorldCropBox(center - half, center + half)
    return {
        "box": box.to_dict(),
        "initial_center": center.tolist(),
        "input_scale": box.scale,
        "margin": float(margin),
    }


def calibrate_directional_crops(
    records: Sequence[EarCalibrationRecord],
    percentile: float = 99.0,
    safety_candidates_mm: Sequence[float] = tuple(range(6)),
    primary_complete_coverage: float = 0.99,
    primary_expansion: float = 0.0,
    backup_expansions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    outward_epsilon_mm: float = 1e-3,
    geometry_stats: Sequence[Mapping[str, float]] = (),
) -> dict:
    if not records:
        raise ValueError("at least one out-of-fold calibration record is required")
    if not 0.0 < float(primary_complete_coverage) <= 1.0:
        raise ValueError("primary complete-ear coverage target must be in (0, 1]")
    if not 0.0 <= float(primary_expansion) <= 1.0:
        raise ValueError("primary expansion must be in [0, 1]")
    true_centers = []
    predictions = []
    landmarks = []
    for record in records:
        points = record.canonical_landmarks()
        landmarks.append(points)
        true_centers.append(ear_bbox_center(points))
        predictions.append(record.canonical_prediction())
    true_centers = np.stack(true_centers)
    predictions = np.stack(predictions)
    minima = np.stack([item.min(axis=0) for item in landmarks])
    maxima = np.stack([item.max(axis=0) for item in landmarks])

    landmark_negative = np.percentile(true_centers - minima, percentile, axis=0)
    landmark_positive = np.percentile(maxima - true_centers, percentile, axis=0)
    error_negative = np.percentile(np.maximum(predictions - true_centers, 0.0), percentile, axis=0)
    error_positive = np.percentile(np.maximum(true_centers - predictions, 0.0), percentile, axis=0)
    base_negative = landmark_negative + error_negative
    base_positive = landmark_positive + error_positive

    selected_safety = None
    selected_coverage = 0.0
    for safety in safety_candidates_mm:
        negative = base_negative + float(safety)
        positive = base_positive + float(safety)
        complete = [
            bool(box_from_center(prediction, negative, positive).contains(points).all())
            for prediction, points in zip(predictions, landmarks)
        ]
        coverage = float(np.mean(complete))
        if coverage >= primary_complete_coverage:
            selected_safety = float(safety)
            selected_coverage = coverage
            break
    if selected_safety is None:
        raise ValueError(
            "safety search did not reach the requested "
            f"{100.0 * float(primary_complete_coverage):g}% complete-ear coverage"
        )

    # Store the reaches in float32 because that is what inference loads.  Moving
    # one representable value outwards prevents JSON/float32 round trips from
    # turning a boundary landmark into a false miss.
    primary_negative = np.nextafter(
        np.asarray(base_negative + selected_safety, dtype=np.float32),
        np.float32(np.inf),
    )
    primary_positive = np.nextafter(
        np.asarray(base_positive + selected_safety, dtype=np.float32),
        np.float32(np.inf),
    )
    if float(primary_expansion) > 0.0:
        factor = 1.0 + float(primary_expansion)
        primary_negative = np.nextafter(
            np.asarray(
                primary_negative * factor + outward_epsilon_mm,
                dtype=np.float32,
            ),
            np.float32(np.inf),
        )
        primary_positive = np.nextafter(
            np.asarray(
                primary_positive * factor + outward_epsilon_mm,
                dtype=np.float32,
            ),
            np.float32(np.inf),
        )
    selected_coverage = _complete_ear_coverage(
        predictions, landmarks, primary_negative, primary_positive
    )

    # The backup must be a true superset of the primary crop.  The previous
    # implementation fitted each backup direction independently to a maximum;
    # this could make some backup axes smaller than their primary counterparts.
    # Select the first proposal-defined expansion that covers every calibration
    # ear, adding a tiny outward tolerance for serialized floating-point bounds.
    selected_backup_expansion = None
    backup_negative = None
    backup_positive = None
    backup_coverage = 0.0
    for expansion in backup_expansions:
        if float(expansion) < 0.0:
            raise ValueError("backup expansions must be nonnegative")
        factor = 1.0 + float(expansion)
        candidate_negative = np.nextafter(
            np.asarray(primary_negative * factor + outward_epsilon_mm, dtype=np.float32),
            np.float32(np.inf),
        )
        candidate_positive = np.nextafter(
            np.asarray(primary_positive * factor + outward_epsilon_mm, dtype=np.float32),
            np.float32(np.inf),
        )
        coverage = _complete_ear_coverage(
            predictions, landmarks, candidate_negative, candidate_positive
        )
        if coverage == 1.0:
            selected_backup_expansion = float(expansion)
            backup_negative = candidate_negative
            backup_positive = candidate_positive
            backup_coverage = coverage
            break
    if selected_backup_expansion is None:
        raise ValueError(
            "backup expansion search did not reach 100% complete-ear coverage"
        )
    result = {
        "schema_version": 1,
        "percentile": float(percentile),
        "primary_complete_coverage_target": float(primary_complete_coverage),
        "primary_expansion": float(primary_expansion),
        "primary_outward_epsilon_mm": (
            float(outward_epsilon_mm) if float(primary_expansion) > 0.0 else 0.0
        ),
        "safety_mm": selected_safety,
        "primary": {
            "negative": primary_negative.astype(np.float32).tolist(),
            "positive": primary_positive.astype(np.float32).tolist(),
            "complete_ear_coverage": selected_coverage,
        },
        "backup": {
            "negative": backup_negative.tolist(),
            "positive": backup_positive.tolist(),
            "complete_ear_coverage": backup_coverage,
            "expansion": selected_backup_expansion,
            "outward_epsilon_mm": float(outward_epsilon_mm),
        },
        "local_scale": float(max(np.max(primary_negative), np.max(primary_positive))),
    }
    valid_geometry = [item for item in geometry_stats if item.get("valid", True)]
    if valid_geometry:
        faces = [float(item["face_count"]) for item in valid_geometry]
        areas = [float(item["surface_area"]) for item in valid_geometry]
        result["fallback_thresholds"] = {
            "face_count_p01": int(np.floor(np.percentile(faces, 1.0))),
            "surface_area_p01": float(np.percentile(areas, 1.0)),
        }
    else:
        result["fallback_thresholds"] = {"face_count_p01": 0, "surface_area_p01": 0.0}
    return result


def _complete_ear_coverage(
    predictions: Sequence[np.ndarray],
    landmarks: Sequence[np.ndarray],
    negative: np.ndarray,
    positive: np.ndarray,
) -> float:
    complete = [
        bool(box_from_center(prediction, negative, positive).contains(points).all())
        for prediction, points in zip(predictions, landmarks)
    ]
    return float(np.mean(complete))


def evaluate_calibration(
    records: Sequence[EarCalibrationRecord], calibration: Mapping[str, object]
) -> dict:
    """Measure serialized crop coverage and report the exact failing ears."""
    if not records:
        raise ValueError("at least one calibration record is required")
    primary = calibration["primary"]
    backup = calibration["backup"]
    primary_misses = []
    backup_misses = []
    center_errors = []
    for record in records:
        points = record.canonical_landmarks()
        prediction = record.canonical_prediction()
        center_errors.append(float(np.linalg.norm(prediction - ear_bbox_center(points))))
        if not box_from_center(
            prediction, primary["negative"], primary["positive"]
        ).contains(points).all():
            primary_misses.append(f"{record.subject_id}:{record.ear}")
        if not box_from_center(
            prediction, backup["negative"], backup["positive"]
        ).contains(points).all():
            backup_misses.append(f"{record.subject_id}:{record.ear}")

    primary_negative = np.asarray(primary["negative"], dtype=np.float64)
    primary_positive = np.asarray(primary["positive"], dtype=np.float64)
    backup_negative = np.asarray(backup["negative"], dtype=np.float64)
    backup_positive = np.asarray(backup["positive"], dtype=np.float64)
    count = len(records)
    return {
        "ear_count": count,
        "center_error_mm": {
            "mean": float(np.mean(center_errors)),
            "median": float(np.median(center_errors)),
            "p95": float(np.percentile(center_errors, 95.0)),
            "p99": float(np.percentile(center_errors, 99.0)),
            "maximum": float(np.max(center_errors)),
        },
        "primary": {
            "complete_count": count - len(primary_misses),
            "coverage": (count - len(primary_misses)) / count,
            "misses": primary_misses,
        },
        "backup": {
            "complete_count": count - len(backup_misses),
            "coverage": (count - len(backup_misses)) / count,
            "misses": backup_misses,
            "is_primary_superset": bool(
                np.all(backup_negative >= primary_negative)
                and np.all(backup_positive >= primary_positive)
            ),
        },
    }


def boxes_for_prediction(predicted_center: np.ndarray, calibration: Mapping[str, object]):
    primary = calibration["primary"]
    backup = calibration["backup"]
    return (
        box_from_center(predicted_center, primary["negative"], primary["positive"]),
        box_from_center(predicted_center, backup["negative"], backup["positive"]),
    )
