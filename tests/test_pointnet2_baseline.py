import numpy as np
import pytest
import torch
import trimesh

from src.dataset import Dataset
from src.ear_crop import (
    CropBox,
    compute_crop_coverage,
    export_subject_crop_plys,
    fit_crop_config_from_training_landmarks,
    points_inside_box,
    sample_crop_point_features,
)
from src.estimator import LandmarkExtractor
from src.metrics import compute_mean_landmark_distance
from src.pointnet2_model import PointNet2LandmarkRegressor, default_model_config
from src.preprocessing import (
    compute_mesh_normalization,
    normalize_point_features,
    sample_mesh_surface,
    split_landmark_prediction,
)
from train_pointnet2 import build_arg_parser, compute_training_loss, resolve_num_landmarks
from visualize_point_importance import colorize_importance, normalize_importance


def _write_landmarks(path, coords):
    with path.open("w", encoding="utf-8") as handle:
        for idx, coord in enumerate(coords):
            handle.write(f"{idx},[{coord[0]} {coord[1]} {coord[2]}]\n")


def _make_landmarks(y_value):
    x = np.linspace(-0.2, 0.2, 85, dtype=np.float32)
    z = np.linspace(-0.1, 0.3, 85, dtype=np.float32)
    y = np.full(85, y_value, dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def _make_tiny_dataset(tmp_path, subject_ids=("S001", "S002")):
    mesh_dir = tmp_path / "mesh"
    landmarks_dir = tmp_path / "landmarks"
    mesh_dir.mkdir()
    landmarks_dir.mkdir()
    mesh = trimesh.creation.box(extents=(2.0, 6.0, 2.0))

    for subject_id in subject_ids:
        mesh.export(mesh_dir / f"{subject_id}.ply")
        _write_landmarks(landmarks_dir / f"{subject_id}_left_ear_landmarks.csv", _make_landmarks(2.0))
        _write_landmarks(
            landmarks_dir / f"{subject_id}_right_ear_landmarks.csv", _make_landmarks(-2.0)
        )

    return mesh_dir, landmarks_dir


def test_surface_sampler_shape_and_normal_lengths():
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    points = sample_mesh_surface(mesh, num_points=128, seed=7)

    assert points.shape == (128, 6)
    normal_lengths = np.linalg.norm(points[:, 3:6], axis=1)
    assert np.allclose(normal_lengths, 1.0, atol=1e-5)


def test_dataset_excludes_malformed_subject_by_default(tmp_path):
    mesh_dir = tmp_path / "mesh"
    landmarks_dir = tmp_path / "landmarks"
    mesh_dir.mkdir()
    landmarks_dir.mkdir()
    (mesh_dir / "P0026.ply").touch()
    (mesh_dir / "P0027.ply").touch()
    (mesh_dir / "P0028.ply").touch()

    dataset = Dataset(mesh_dir=str(mesh_dir), landmarks_dir=str(landmarks_dir))

    assert dataset.subject_ids == ["P0026", "P0028"]


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


def test_single_ear_crop_model_forward_shape():
    model = PointNet2LandmarkRegressor(
        num_landmarks=85,
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

    assert output.shape == (2, 85, 3)


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


def test_mean_distance_loss_matches_official_metric_after_denormalization():
    pred_normalized = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, -0.5, 1.0]]], dtype=torch.float32
    )
    target_normalized = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.25, -0.25, 0.5]]], dtype=torch.float32
    )
    centroid = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)
    scale = torch.tensor([2.0], dtype=torch.float32)

    loss = compute_training_loss(
        None,
        "mean_distance",
        pred_normalized,
        target_normalized,
        centroid,
        scale,
    )
    pred = pred_normalized.numpy().reshape(2, 3) * 2.0 + np.array([10.0, 20.0, 30.0])
    target = target_normalized.numpy().reshape(2, 3) * 2.0 + np.array([10.0, 20.0, 30.0])

    assert np.isclose(loss.item(), compute_mean_landmark_distance(pred, target))


def test_normalize_point_features_keeps_normals_unit_length():
    mesh = trimesh.creation.box()
    transform = compute_mesh_normalization(mesh)
    points = sample_mesh_surface(mesh, num_points=16, seed=3)
    points[:, 3:6] *= 3.0

    normalized = normalize_point_features(points, transform)

    assert np.allclose(np.linalg.norm(normalized[:, 3:6], axis=1), 1.0, atol=1e-5)


def test_crop_boxes_are_fit_from_training_subjects_only(tmp_path):
    mesh_dir, landmarks_dir = _make_tiny_dataset(tmp_path, subject_ids=("TRAIN", "VAL"))
    dataset = Dataset(str(mesh_dir), str(landmarks_dir))

    crop_config = fit_crop_config_from_training_landmarks(dataset, ["TRAIN"], margin=0.0)
    left_box = crop_config["left"]
    outside_training_range = np.array([[0.0, 20.0, 0.0]], dtype=np.float32)

    assert "left" in crop_config
    assert left_box.minimum.shape == (3,)
    assert not np.all(
        (outside_training_range[0] >= left_box.minimum)
        & (outside_training_range[0] <= left_box.maximum)
    )


def test_crop_coverage_reports_all_landmarks_inside(tmp_path):
    mesh_dir, landmarks_dir = _make_tiny_dataset(tmp_path)
    dataset = Dataset(str(mesh_dir), str(landmarks_dir))
    subject_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
    crop_config = fit_crop_config_from_training_landmarks(dataset, subject_ids, margin=0.4)

    coverage = compute_crop_coverage(dataset, subject_ids, crop_config)

    assert coverage["summary"]["left"]["coverage"] == 1.0
    assert coverage["summary"]["right"]["coverage"] == 1.0


def test_crop_dataset_returns_single_ear_samples(tmp_path):
    mesh_dir, landmarks_dir = _make_tiny_dataset(tmp_path)
    from src.torch_dataset import PinnaEarCropDataset

    dataset = Dataset(str(mesh_dir), str(landmarks_dir))
    subject_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
    crop_config = {
        "left": CropBox(
            minimum=np.array([-1.0, 0.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        ),
        "right": CropBox(
            minimum=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        ),
    }
    crop_dataset = PinnaEarCropDataset(
        str(mesh_dir),
        str(landmarks_dir),
        crop_config=crop_config,
        ear_points=32,
        subject_ids=subject_ids,
    )

    item = crop_dataset[0]

    assert len(crop_dataset) == len(subject_ids) * 2
    assert item["points"].shape == (32, 6)
    assert item["landmarks"].shape == (85, 3)
    assert item["ear"] == "left"


def test_crop_sampler_keeps_points_inside_box():
    mesh = trimesh.creation.box(extents=(2.0, 6.0, 2.0))
    transform = compute_mesh_normalization(mesh)
    crop_box = CropBox(
        minimum=np.array([-0.35, 0.25, -0.35], dtype=np.float32),
        maximum=np.array([0.35, 0.75, 0.35], dtype=np.float32),
    )

    sampled = sample_crop_point_features(
        mesh,
        transform,
        crop_box,
        32,
        seed=13,
        oversample_factor=64,
        max_attempts=3,
    )

    assert sampled.shape == (32, 6)
    assert np.all(sampled[:, :3] >= crop_box.minimum - 1e-5)
    assert np.all(sampled[:, :3] <= crop_box.maximum + 1e-5)


def test_crop_export_writes_exact_sampled_point_cloud(tmp_path):
    mesh_dir, landmarks_dir = _make_tiny_dataset(tmp_path, subject_ids=("S001",))
    dataset = Dataset(str(mesh_dir), str(landmarks_dir))
    subject_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
    crop_config = {
        "left": CropBox(
            minimum=np.array([-1.0, 0.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        ),
        "right": CropBox(
            minimum=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        ),
    }
    output_dir = tmp_path / "outputs"

    stats = export_subject_crop_plys(
        dataset,
        subject_ids,
        crop_config,
        str(output_dir),
        "train",
        ear_points=16,
        oversample_factor=32,
        max_attempts=3,
    )

    mesh_path = output_dir / "crops" / "train" / "S001_left_mesh.ply"
    points_path = output_dir / "crops" / "train" / "S001_left_points.ply"
    point_cloud = trimesh.load(points_path, process=False)

    assert mesh_path.exists()
    assert points_path.exists()
    assert point_cloud.vertices.shape[0] == 16
    assert stats["S001"]["left"]["final_sampled_count"] == 16


def test_right_ear_mirroring_flips_y_and_normal_y():
    mesh = trimesh.creation.box(extents=(2.0, 6.0, 2.0))
    transform = compute_mesh_normalization(mesh)
    crop_box = CropBox(
        minimum=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
        maximum=np.array([2.0, 2.0, 2.0], dtype=np.float32),
    )

    original = sample_crop_point_features(mesh, transform, crop_box, 32, seed=11, mirror_y=False)
    mirrored = sample_crop_point_features(mesh, transform, crop_box, 32, seed=11, mirror_y=True)

    assert np.allclose(mirrored[:, 0], original[:, 0])
    assert np.allclose(mirrored[:, 1], -original[:, 1])
    assert np.allclose(mirrored[:, 2:4], original[:, 2:4])
    assert np.allclose(mirrored[:, 4], -original[:, 4])
    assert np.allclose(mirrored[:, 5], original[:, 5])


def test_ear_crop_arg_defaults_use_single_ear_landmarks_and_no_mirroring():
    parser = build_arg_parser()
    args = parser.parse_args(["--input-mode", "ear_crop"])

    resolve_num_landmarks(args)

    assert args.num_landmarks == 85
    assert args.mirror_right_ear is False


def test_ear_crop_rejects_incompatible_landmark_count():
    parser = build_arg_parser()
    args = parser.parse_args(["--input-mode", "ear_crop", "--num-landmarks", "170"])

    with pytest.raises(ValueError, match="requires --num-landmarks 85"):
        resolve_num_landmarks(args)


def test_estimator_ear_crop_checkpoint_returns_two_single_ear_predictions(tmp_path):
    mesh = trimesh.creation.box(extents=(2.0, 6.0, 2.0))
    model_config = default_model_config()
    model_config.update(
        {
            "num_landmarks": 85,
            "dropout": 0.0,
            "head_channels": [32],
            "ssg_npoints": [16, 4],
            "ssg_radii": [0.4, 0.8],
            "ssg_nsamples": [8, 8],
            "ssg_mlps": [[8, 8, 16], [16, 16, 32], [32, 64]],
        }
    )
    model = PointNet2LandmarkRegressor(**model_config)
    crop_config = {
        "left": CropBox(
            minimum=np.array([-1.0, 0.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        ).to_dict(),
        "right": CropBox(
            minimum=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        ).to_dict(),
    }
    checkpoint_path = tmp_path / "single_ear_crop.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "input_mode": "ear_crop",
            "crop_config": crop_config,
            "ear_points": 32,
            "mirror_right_ear": False,
            "seed": 3,
        },
        checkpoint_path,
    )

    extractor = LandmarkExtractor(checkpoint_path=str(checkpoint_path))
    left, right = extractor.extract(mesh)

    assert left.shape == (85, 3)
    assert right.shape == (85, 3)


def test_importance_normalization_handles_constant_and_finite_values():
    constant = normalize_importance(np.array([5.0, 5.0, 5.0], dtype=np.float32))
    varied = normalize_importance(np.array([2.0, 4.0, 6.0], dtype=np.float32))

    assert np.allclose(constant, 0.0)
    assert np.allclose(varied, np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_importance_colorize_outputs_rgba_uint8():
    colors = colorize_importance(np.array([0.0, 0.5, 1.0], dtype=np.float32))

    assert colors.shape == (3, 4)
    assert colors.dtype == np.uint8
    assert np.all(colors[:, 3] == 255)
