import copy

import numpy as np
import pytest
import torch

from src.proposal_models import LocalLandmarkRefiner, build_landmark_model
from src.surface_geometry_features import (
    append_surface_geometry_features,
    make_surface_geometry_config,
    validate_surface_geometry_config,
)
from train_pipeline import build_parser, landmark_model_config


def _plane_features(side=20):
    axis = np.linspace(-0.08, 0.08, side, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    xyz = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    normals = np.repeat([[0.0, 0.0, 1.0]], len(xyz), axis=0)
    return np.concatenate([xyz, normals], axis=1).astype(np.float32)


def test_disabled_geometry_preserves_six_channels_bit_for_bit():
    points = _plane_features(8)
    output = append_surface_geometry_features(
        points, 40.0, make_surface_geometry_config(enabled=False)
    )
    assert output.shape == points.shape
    assert np.array_equal(output, points)
    assert not np.shares_memory(output, points)


def test_planar_geometry_is_finite_deterministic_and_flat():
    points = _plane_features()
    config = make_surface_geometry_config(enabled=True)
    first = append_surface_geometry_features(points, 40.0, config)
    second = append_surface_geometry_features(points, 40.0, config)
    assert first.shape == (len(points), 14)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.max(np.abs(first[:, 6:13])) < 1e-6
    assert np.allclose(first[:, 13], np.linalg.norm(points[:, :3], axis=1))


def test_geometry_channels_are_invariant_to_right_style_reflection():
    points = _plane_features()
    points[:, 2] = 0.15 * (points[:, 0] ** 2 + 0.5 * points[:, 1] ** 2)
    normals = np.stack(
        [-0.3 * points[:, 0], -0.15 * points[:, 1], np.ones(len(points))],
        axis=1,
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    points[:, 3:6] = normals
    reflected = points.copy()
    reflected[:, 1] *= -1.0
    reflected[:, 4] *= -1.0
    config = make_surface_geometry_config(enabled=True)
    left = append_surface_geometry_features(points, 40.0, config)
    right = append_surface_geometry_features(reflected, 40.0, config)
    assert np.allclose(left[:, 6:], right[:, 6:], atol=2e-5, rtol=2e-5)


def test_signed_gaussian_curvature_separates_a_saddle():
    points = _plane_features(21)
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    points[:, 2] = 3.0 * (x * x - y * y)
    normals = np.stack([-6.0 * x, 6.0 * y, np.ones(len(points))], axis=1)
    points[:, 3:6] = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    output = append_surface_geometry_features(
        points, 40.0, make_surface_geometry_config(enabled=True)
    )
    centre = np.argmin(np.linalg.norm(points[:, :2], axis=1))
    assert output[centre, 10] < -1e-3


def test_surface_geometry_config_rejects_schema_drift():
    config = make_surface_geometry_config(enabled=True)
    broken = copy.deepcopy(config)
    broken["feature_names"] = list(reversed(broken["feature_names"]))
    with pytest.raises(ValueError, match="names/order"):
        validate_surface_geometry_config(broken)
    broken = copy.deepcopy(config)
    broken["output_channels"] = 13
    with pytest.raises(ValueError, match="output channel"):
        validate_surface_geometry_config(broken)


def test_model_rejects_geometry_channels_without_saved_preprocessing():
    args = _geometry_args()
    config = landmark_model_config(args, local_scale=40.0)
    del config["surface_geometry"]
    with pytest.raises(ValueError, match="saved preprocessing"):
        build_landmark_model(config)


def _geometry_args():
    return build_parser().parse_args(
        [
            "fit-landmarks",
            "--folds-json", "folds.json",
            "--outer-fold", "0",
            "--predictions-json", "predictions.json",
            "--calibration-json", "calibration.json",
            "--output-dir", "runs/geometry",
            "--backbone", "pointnext",
            "--pointnext-variant", "s",
            "--landmark-decoder", "surface-heatmap",
            "--heatmap-feature-dim", "32",
            "--heatmap-weight", "0.1",
            "--surface-geometry-features",
            "--surface-geometry-radii-mm", "1", "2", "4",
            "--surface-geometry-neighbours", "64",
            "--surface-curvature-radius-mm", "2",
            "--refinement-k", "32",
            "--no-augment",
        ]
    )


def test_cli_records_geometry_schema_and_model_accepts_fourteen_channels():
    args = _geometry_args()
    config = landmark_model_config(args, local_scale=40.0)
    assert config["encoder_config"]["input_channels"] == 14
    assert config["surface_geometry"]["feature_names"] == [
        "normal_variation_r1mm",
        "normal_variation_r2mm",
        "normal_variation_r4mm",
        "bounded_abs_mean_curvature",
        "bounded_signed_gaussian_curvature",
        "abs_shape_index",
        "bounded_curvedness",
        "crop_centre_distance",
    ]
    model = build_landmark_model(config).eval()
    points = torch.randn(1, 64, 14)
    points[..., 3:6] = torch.nn.functional.normalize(points[..., 3:6], dim=-1)
    with torch.no_grad():
        details = model.forward_with_details(points)
    assert details["final"].shape == (1, 85, 3)
    assert torch.isfinite(details["final"]).all()


def test_geometry_experiment_rejects_confounded_modes():
    args = _geometry_args()
    args.augment = True
    with pytest.raises(ValueError, match="no-augment"):
        landmark_model_config(args, local_scale=40.0)
    args = _geometry_args()
    args.heatmap_distance = "geodesic"
    with pytest.raises(ValueError, match="Euclidean"):
        landmark_model_config(args, local_scale=40.0)


def test_geometry_offset_refiner_ignores_extra_channels_safely():
    refiner = LocalLandmarkRefiner(32, 0.1).eval()
    points = torch.randn(2, 64, 14)
    points[..., 3:6] = torch.nn.functional.normalize(points[..., 3:6], dim=-1)
    coarse = torch.randn(2, 85, 3)
    with torch.no_grad():
        output = refiner(coarse, points)
    assert output.shape == coarse.shape
    assert torch.isfinite(output).all()
