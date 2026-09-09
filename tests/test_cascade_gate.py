import numpy as np
import pytest

from src.shape_prior.evaluate_cascade_gate import (
    GATE_FEATURES,
    calibrate_gate_threshold,
    gate_alpha,
)
from train_pipeline import build_parser


def test_ear_gate_uses_unlabelled_training_distribution_and_broadcasts():
    training = np.stack(
        [
            np.full(85, 0.0),
            np.full(85, 1.0),
            np.full(85, 2.0),
            np.full(85, 3.0),
        ]
    )
    threshold = calibrate_gate_threshold(training, "ear", 0.5)
    assert threshold == pytest.approx(1.5)
    validation = np.stack([np.full(85, 1.0), np.full(85, 2.0)])
    alpha = gate_alpha(validation, threshold, "ear", 0.5)
    assert alpha.shape == (2, 85)
    assert np.all(alpha[0] == 0.0)
    assert np.all(alpha[1] == 0.5)


def test_landmark_gate_calibrates_each_identity_independently():
    base = np.arange(85, dtype=np.float64)
    training = np.stack([base, base + 1.0, base + 2.0])
    threshold = calibrate_gate_threshold(training, "landmark", 0.5)
    assert threshold.shape == (85,)
    np.testing.assert_allclose(threshold, base + 1.0)
    validation = np.stack([base + 0.5, base + 1.5])
    alpha = gate_alpha(validation, threshold, "landmark", 1.0)
    assert np.all(alpha[0] == 0.0)
    assert np.all(alpha[1] == 1.0)


def test_gate_rejects_invalid_shapes_scopes_and_blends():
    with pytest.raises(ValueError, match="shape"):
        calibrate_gate_threshold(np.ones((3, 84)), "ear", 0.5)
    with pytest.raises(ValueError, match="unknown gate scope"):
        calibrate_gate_threshold(np.ones((3, 85)), "subject", 0.5)
    with pytest.raises(ValueError, match="blend"):
        gate_alpha(np.ones((2, 85)), 0.5, "ear", 0.0)
    with pytest.raises(ValueError, match="shape"):
        gate_alpha(np.ones((2, 85)), np.ones(84), "landmark", 0.5)


def test_pipeline_parser_exposes_fixed_and_screening_gate_controls():
    args = build_parser().parse_args(
        [
            "evaluate-cascade-gate",
            "--baseline-checkpoint-path",
            "baseline.pt",
            "--cascade-checkpoint-path",
            "cascade.pt",
            "--prior-path",
            "prior.npz",
            "--prior-manifest",
            "manifest.json",
            "--folds-json",
            "folds.json",
            "--predictions-json",
            "predictions.json",
            "--calibration-json",
            "calibration.json",
            "--gate-features",
            "entropy",
            "model_disagreement_mm",
            "--gate-scopes",
            "ear",
            "--gate-quantiles",
            "0.8",
            "--gate-blends",
            "0.5",
            "--output",
            "report.json",
        ]
    )
    assert args.gate_features == ["entropy", "model_disagreement_mm"]
    assert args.gate_scopes == ["ear"]
    assert args.gate_quantiles == [pytest.approx(0.8)]
    assert args.gate_blends == [pytest.approx(0.5)]
    assert args.projection_workers == 10
    assert set(GATE_FEATURES).issuperset(args.gate_features)
