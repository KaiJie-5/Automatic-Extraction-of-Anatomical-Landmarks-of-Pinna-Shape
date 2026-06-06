import numpy as np
import pytest
import torch
import trimesh

from src.estimator import LandmarkExtractor
from src.metrics import compute_mean_landmark_distance
from src.pointnet2_model import PointNet2LandmarkRegressor
from src.preprocessing import (
    compute_mesh_normalization,
    normalize_point_features,
    sample_mesh_surface,
    split_landmark_prediction,
)


def test_surface_sampler_shape_and_normal_lengths():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    points = sample_mesh_surface(mesh, num_points=128, seed=7)

    assert points.shape == (128, 6)
    normal_lengths = np.linalg.norm(points[:, 3:6], axis=1)
    assert np.allclose(normal_lengths, 1.0, atol=1e-5)


def test_normalization_round_trip():
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    transform = compute_mesh_normalization(mesh)
    xyz = np.array([[0.25, -0.5, 1.5], [1.0, 2.0, -3.0]], dtype=np.float32)

    restored = transform.denormalize_xyz(transform.normalize_xyz(xyz))

    assert np.allclose(restored, xyz, atol=1e-6)


def test_pointnet2_ssg_forward_shape():
    model = PointNet2LandmarkRegressor(
        num_landmarks=170,
        variant="ssg",
        dropout=0.0,
        head_channels=[32],
        ssg_npoints=[16, 4],
        ssg_radii=[0.4, 0.8],
        ssg_nsamples=[8, 8],
        ssg_mlps=[[8, 8, 16], [16, 16, 32], [32, 64]],
    )
    model.eval()
    points = torch.randn(2, 64, 6)

    with torch.no_grad():
        output = model(points)

    assert output.shape == (2, 170, 3)


def test_pointnet2_msg_forward_shape():
    model = PointNet2LandmarkRegressor(
        num_landmarks=170,
        variant="msg",
        dropout=0.0,
        head_channels=[32],
        msg_npoints=[16, 4],
        msg_radii=[[0.2, 0.4], [0.4, 0.8]],
        msg_nsamples=[[8, 16], [8, 16]],
        msg_mlps=[[[8, 8], [8, 16]], [[16, 16], [16, 32]]],
        msg_global_mlp=[32, 64],
    )
    model.eval()
    points = torch.randn(2, 64, 6)

    with torch.no_grad():
        output = model(points)

    assert output.shape == (2, 170, 3)


def test_estimator_missing_checkpoint_raises(tmp_path):
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="Missing trained checkpoint"):
        LandmarkExtractor(checkpoint_path=str(missing_path))


def test_split_metric_matches_left_right_average():
    prediction = np.arange(170 * 3, dtype=np.float32).reshape(170, 3)
    target = prediction + 1.0
    pred_left, pred_right = split_landmark_prediction(prediction)
    target_left, target_right = split_landmark_prediction(target)

    split_score = (
        compute_mean_landmark_distance(pred_left, target_left)
        + compute_mean_landmark_distance(pred_right, target_right)
    ) / 2.0
    combined_score = compute_mean_landmark_distance(prediction, target)

    assert np.isclose(split_score, combined_score)


def test_normalize_point_features_keeps_normals_unit_length():
    mesh = trimesh.creation.box()
    transform = compute_mesh_normalization(mesh)
    points = sample_mesh_surface(mesh, num_points=16, seed=3)
    points[:, 3:6] *= 3.0

    normalized = normalize_point_features(points, transform)

    assert np.allclose(np.linalg.norm(normalized[:, 3:6], axis=1), 1.0, atol=1e-5)
