import json

import numpy as np
import pytest
import torch
import trimesh

from src.audit import audit_dataset
from src.calibration import (
    EarCalibrationRecord,
    calibrate_directional_crops,
    evaluate_calibration,
)
from src.canonical import (
    LocalEarTransform,
    WorldCropBox,
    canonicalize_point_features,
    canonicalize_xyz,
    decanonicalize_xyz,
    ear_bbox_center,
)
from src.losses import (
    ANCHOR_INDICES,
    SPACING_SECTIONS,
    candidate_is_promoted,
    proposal_landmark_loss,
)
from src.meshnet import validate_meshnet_mesh
from src.estimator import LandmarkExtractor
from src.pipeline_dataset import EpochResampledDataset
from src.pointnext_model import PointNeXtEncoder, default_pointnext_config
from src.proposal_models import ProposalLandmarkRegressor
from src.splits import make_nested_folds
from src.surface import project_points_to_mesh
from train_pipeline import build_parser, landmark_model_config


def _write_landmarks(path, points):
    with path.open("w", encoding="utf-8") as handle:
        for index, point in enumerate(points):
            handle.write(f"{index},[{point[0]} {point[1]} {point[2]}]\n")


def _valid_data_root(tmp_path):
    mesh_dir = tmp_path / "mesh"
    landmark_dir = tmp_path / "landmarks"
    mesh_dir.mkdir()
    landmark_dir.mkdir()
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=10.0)
    points = np.linspace(-1.0, 1.0, 85 * 3, dtype=np.float32).reshape(85, 3)
    for subject in ("KEMAR", "P0027"):
        mesh.export(mesh_dir / f"{subject}.ply")
        _write_landmarks(landmark_dir / f"{subject}_left_ear_landmarks.csv", points)
        _write_landmarks(landmark_dir / f"{subject}_right_ear_landmarks.csv", points)
    return mesh_dir, landmark_dir


def test_audit_includes_kemar_and_corrected_p0027(tmp_path):
    mesh_dir, landmark_dir = _valid_data_root(tmp_path)
    report = audit_dataset(str(mesh_dir), str(landmark_dir), fail_on_fatal=True)
    assert report["summary"]["complete_subjects"] == 2
    assert report["summary"]["complete_ears"] == 4
    assert report["summary"]["kemar_included"]
    assert report["summary"]["p0027_included"]


def test_audit_writes_consolidated_report_before_raising(tmp_path):
    mesh_dir, landmark_dir = _valid_data_root(tmp_path)
    (landmark_dir / "P0027_right_ear_landmarks.csv").unlink()
    output = tmp_path / "analysis"
    with pytest.raises(ValueError, match="fatal issue"):
        audit_dataset(str(mesh_dir), str(landmark_dir), str(output), fail_on_fatal=True)
    reports = list(output.glob("audit_*/dataset_audit.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["summary"]["fatal_issues"] > 0


def test_nested_folds_are_deterministic_disjoint_and_balanced():
    metrics = {
        f"S{index:03d}": {"y_extent": 100.0 + index, "vertices": 1000 + index * 17}
        for index in range(201)
    }
    first = make_nested_folds(metrics, seed=42)
    second = make_nested_folds(metrics, seed=42)
    assert first == second
    assert [len(item["validation"]) for item in first["outer"]] == [41, 40, 40, 40, 40]
    for outer in first["outer"]:
        assert set(outer["train"]).isdisjoint(outer["validation"])
        assert set(outer["train"]) | set(outer["validation"]) == set(metrics)
        for inner in outer["inner"]:
            assert set(inner["train"]).isdisjoint(inner["validation"])
            assert set(inner["train"]) | set(inner["validation"]) == set(outer["train"])


def test_right_canonical_transform_and_local_transform_round_trip():
    xyz = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)
    features = np.concatenate([xyz, xyz / np.linalg.norm(xyz, axis=1, keepdims=True)], axis=1)
    mirrored = canonicalize_point_features(features, "right")
    assert np.allclose(mirrored[:, 1], -features[:, 1])
    assert np.allclose(mirrored[:, 4], -features[:, 4])
    assert np.allclose(decanonicalize_xyz(canonicalize_xyz(xyz, "right"), "right"), xyz)
    local = LocalEarTransform(np.array([10.0, 20.0, 30.0]), 40.0)
    assert np.allclose(local.denormalize_xyz(local.normalize_xyz(xyz)), xyz)


def test_bbox_center_and_directional_signed_error_formula():
    axis = np.linspace(-1.0, 1.0, 85, dtype=np.float32)
    landmarks = np.stack([axis, axis * 2.0, axis * 3.0], axis=1)
    prediction = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    records = [EarCalibrationRecord(f"S{i}", "left", landmarks, prediction) for i in range(8)]
    calibration = calibrate_directional_crops(records)
    assert np.allclose(ear_bbox_center(landmarks), 0.0)
    assert np.allclose(calibration["primary"]["negative"], [1.5, 2.0, 4.0])
    assert np.allclose(calibration["primary"]["positive"], [1.0, 2.5, 3.0])
    assert calibration["primary"]["complete_ear_coverage"] == 1.0
    assert calibration["backup"]["complete_ear_coverage"] == 1.0
    assert calibration["backup"]["expansion"] == 0.2
    assert np.all(
        np.asarray(calibration["backup"]["negative"])
        >= np.asarray(calibration["primary"]["negative"])
    )
    assert np.all(
        np.asarray(calibration["backup"]["positive"])
        >= np.asarray(calibration["primary"]["positive"])
    )
    report = evaluate_calibration(records, calibration)
    assert report["primary"]["coverage"] == 1.0
    assert report["backup"]["coverage"] == 1.0
    assert report["backup"]["is_primary_superset"]


def test_backup_crop_search_rejects_insufficient_expansions():
    axis = np.linspace(-1.0, 1.0, 85, dtype=np.float32)
    normal = np.stack([axis, axis, axis], axis=1)
    outlier = normal.copy()
    outlier[0, 0] = -100.0
    records = [
        EarCalibrationRecord(f"S{i}", "left", normal, np.zeros(3, dtype=np.float32))
        for i in range(100)
    ]
    records.append(
        EarCalibrationRecord("outlier", "left", outlier, np.zeros(3, dtype=np.float32))
    )
    with pytest.raises(ValueError, match="backup expansion search"):
        calibrate_directional_crops(records, backup_expansions=(0.0,))


def test_primary_complete_coverage_target_selects_robust_safety_margin():
    axis = np.linspace(-1.0, 1.0, 85, dtype=np.float32)
    normal = np.stack([axis, axis, axis], axis=1)
    outlier = normal.copy()
    outlier[0, 0] = -3.0
    records = [
        EarCalibrationRecord(f"S{i}", "left", normal, np.zeros(3, dtype=np.float32))
        for i in range(100)
    ]
    records.append(
        EarCalibrationRecord("outlier", "left", outlier, np.zeros(3, dtype=np.float32))
    )

    calibration = calibrate_directional_crops(
        records,
        primary_complete_coverage=1.0,
    )

    assert calibration["primary_complete_coverage_target"] == 1.0
    assert calibration["safety_mm"] == 2.0
    assert calibration["primary"]["complete_ear_coverage"] == 1.0

    expanded = calibrate_directional_crops(
        records,
        primary_complete_coverage=1.0,
        primary_expansion=0.2,
    )
    assert expanded["primary_expansion"] == 0.2
    assert expanded["primary_outward_epsilon_mm"] == pytest.approx(1e-3)
    assert np.all(
        np.asarray(expanded["primary"]["negative"])
        > np.asarray(calibration["primary"]["negative"])
    )
    assert np.all(
        np.asarray(expanded["primary"]["positive"])
        > np.asarray(calibration["primary"]["positive"])
    )
    assert expanded["local_scale"] > calibration["local_scale"]


def test_calibrate_cli_accepts_primary_complete_coverage_target():
    args = build_parser().parse_args(
        [
            "calibrate",
            "--locator-run-root",
            "runs/locator_cv",
            "--primary-complete-coverage",
            "1.0",
            "--primary-expansion",
            "0.2",
            "--output",
            "artifacts/calibration.json",
        ]
    )
    assert args.primary_complete_coverage == 1.0
    assert args.primary_expansion == 0.2


def test_epoch_sampling_changes_only_for_dynamic_dataset():
    dynamic = EpochResampledDataset(seed=42, dynamic_sampling=True)
    fixed = EpochResampledDataset(seed=42, dynamic_sampling=False)
    before = dynamic.sample_seed(3)
    dynamic.set_epoch(2)
    fixed.set_epoch(2)
    assert dynamic.sample_seed(3) != before
    assert fixed.sample_seed(3) == 42 + 3 * 1009


def _tiny_pointnet_config():
    return {
        "input_channels": 6,
        "use_normals": True,
        "variant": "ssg",
        "ssg_npoints": [16, 4],
        "ssg_radii": [0.4, 0.8],
        "ssg_nsamples": [8, 8],
        "ssg_mlps": [[8, 8, 16], [16, 16, 32], [32, 64]],
    }


def test_four_contour_heads_preserve_85_landmark_order_shape():
    model = ProposalLandmarkRegressor(
        backbone="pointnet2",
        encoder_config=_tiny_pointnet_config(),
        four_heads=True,
        head_channels=[32],
    ).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 64, 6))
    assert output.shape == (2, 85, 3)


def test_local_refiner_shapes_for_both_knn_sizes():
    for k in (32, 64):
        model = ProposalLandmarkRegressor(
            backbone="pointnet2",
            encoder_config=_tiny_pointnet_config(),
            four_heads=False,
            head_channels=[32],
            refinement_k=k,
            refinement_cap_normalized=0.1,
        ).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 64, 6))
        assert output.shape == (1, 85, 3)


def test_pointnext_portable_forward_shape():
    encoder = PointNeXtEncoder(
        input_channels=6, width=8, strides=[1, 2, 2, 2, 2], blocks=[1, 1, 1, 1, 1], nsample=8
    ).eval()
    with torch.no_grad():
        output = encoder(torch.randn(2, 64, 6))
    assert output.shape == (2, 8 * 16)


def test_pointnext_width_is_cli_tunable_and_checkpoint_ready():
    parser = build_parser()
    common = [
        "fit-landmarks",
        "--folds-json",
        "folds.json",
        "--outer-fold",
        "0",
        "--predictions-json",
        "predictions.json",
        "--calibration-json",
        "calibration.json",
        "--output-dir",
        "runs/pointnext",
        "--backbone",
        "pointnext",
    ]
    default_args = parser.parse_args(common)
    c64_args = parser.parse_args([*common, "--pointnext-width", "64"])
    final_args = parser.parse_args(
        [
            "fit-final",
            "--locator-run-root",
            "runs/locator_cv",
            "--calibration-json",
            "calibration.json",
            "--locator-epochs",
            "100",
            "--landmark-epochs",
            "100",
            "--backbone",
            "pointnext",
            "--pointnext-width",
            "64",
        ]
    )

    default_config = landmark_model_config(default_args, local_scale=40.0)
    c64_config = landmark_model_config(c64_args, local_scale=40.0)
    final_config = landmark_model_config(final_args, local_scale=40.0)
    assert default_args.pointnext_width == 32
    assert default_config["encoder_config"]["width"] == 32
    assert c64_args.pointnext_width == 64
    assert c64_config["encoder_config"]["width"] == 64
    assert final_config["encoder_config"]["width"] == 64
    assert PointNeXtEncoder(**c64_config["encoder_config"]).feature_dim == 1024


def test_pointnext_width_rejects_nonpositive_values():
    assert default_pointnext_config()["width"] == 32
    assert default_pointnext_config(64)["width"] == 64
    with pytest.raises(ValueError, match="positive integer"):
        default_pointnext_config(0)

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "fit-landmarks",
                "--folds-json",
                "folds.json",
                "--outer-fold",
                "0",
                "--predictions-json",
                "predictions.json",
                "--calibration-json",
                "calibration.json",
                "--output-dir",
                "runs/invalid",
                "--backbone",
                "pointnext",
                "--pointnext-width",
                "0",
            ]
        )


def test_pointnext_b_variant_is_cli_tunable_and_checkpoint_ready():
    parser = build_parser()
    common = [
        "fit-landmarks",
        "--folds-json",
        "folds.json",
        "--outer-fold",
        "0",
        "--predictions-json",
        "predictions.json",
        "--calibration-json",
        "calibration.json",
        "--output-dir",
        "runs/pointnext_b",
        "--backbone",
        "pointnext",
        "--pointnext-variant",
        "b",
    ]
    args = parser.parse_args(common)
    final_args = parser.parse_args(
        [
            "fit-final",
            "--locator-run-root",
            "runs/locator_cv",
            "--calibration-json",
            "calibration.json",
            "--locator-epochs",
            "100",
            "--landmark-epochs",
            "100",
            "--backbone",
            "pointnext",
            "--pointnext-variant",
            "b",
            "--pointnext-width",
            "32",
        ]
    )
    config = landmark_model_config(args, local_scale=40.0)
    final_config = landmark_model_config(final_args, local_scale=40.0)
    encoder_config = config["encoder_config"]

    assert args.pointnext_variant == "b"
    assert encoder_config["variant"] == "b"
    assert encoder_config["width"] == 32
    assert encoder_config["blocks"] == [1, 2, 3, 2, 2]
    assert final_config["encoder_config"]["variant"] == "b"
    assert final_config["encoder_config"]["blocks"] == [1, 2, 3, 2, 2]

    encoder = PointNeXtEncoder(
        **{
            **encoder_config,
            "width": 8,
            "strides": [1, 2, 2, 2, 2],
            "nsample": 8,
        }
    ).train()
    assert sum(len(stage) for stage in encoder.residual_stages) == 5
    points = torch.randn(1, 64, 6, requires_grad=True)
    output = encoder(points)
    assert output.shape == (1, 8 * 16)
    output.square().mean().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
        if parameter.requires_grad
    )


def test_pointnext_variant_config_validation():
    assert default_pointnext_config(variant="s")["blocks"] == [1, 1, 1, 1, 1]
    assert default_pointnext_config(variant="b")["blocks"] == [1, 2, 3, 2, 2]
    with pytest.raises(ValueError, match="unsupported PointNeXt variant"):
        default_pointnext_config(variant="xl")
    with pytest.raises(ValueError, match="requires blocks"):
        PointNeXtEncoder(variant="b", blocks=[1, 1, 1, 1, 1])


def test_pointnext_s_variant_preserves_legacy_checkpoint_keys():
    common = {
        "input_channels": 6,
        "width": 8,
        "strides": [1, 2, 2, 2, 2],
        "blocks": [1, 1, 1, 1, 1],
        "nsample": 8,
    }
    legacy = PointNeXtEncoder(**common)
    current = PointNeXtEncoder(**common, variant="s")
    assert set(legacy.state_dict()) == set(current.state_dict())
    current.load_state_dict(legacy.state_dict(), strict=True)


def test_proposal_losses_have_expected_zero_and_positive_terms():
    target = torch.zeros(2, 85, 3)
    prediction = target.clone()
    surface = torch.zeros(2, 100, 3)
    losses = proposal_landmark_loss(
        prediction,
        target,
        torch.tensor([10.0, 10.0]),
        dense_surface=surface,
        anchor_weight=0.1,
        spacing_weight=0.1,
        surface_weight=0.1,
    )
    assert len(ANCHOR_INDICES) == 15
    assert len(SPACING_SECTIONS) == 11
    assert losses["total"].item() == 0.0


def test_promotion_requires_pooled_and_three_fold_improvement():
    baseline = [2.0] * 5
    assert candidate_is_promoted(baseline, [1.9, 1.9, 1.9, 2.1, 2.1], 20.0, 30.0)
    assert not candidate_is_promoted(baseline, [1.8, 1.8, 2.1, 2.1, 2.1], 20.0, 10.0)


def test_exact_triangle_projection_and_meshnet_open_mesh_gate():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    projected = project_points_to_mesh(np.array([[0.25, 0.25, 2.0]], dtype=np.float32), mesh)
    assert np.allclose(projected, [[0.25, 0.25, 0.0]])
    valid, reason = validate_meshnet_mesh(mesh)
    assert valid, reason
    assert not mesh.is_watertight


def test_world_crop_reflection_swaps_y_bounds():
    box = WorldCropBox(np.array([-1.0, 2.0, -3.0]), np.array([4.0, 6.0, 7.0]))
    right = box.for_ear("right")
    assert np.allclose(right.minimum, [-1.0, -6.0, -3.0])
    assert np.allclose(right.maximum, [4.0, -2.0, 7.0])


def test_v2_estimator_is_deterministic_and_returns_both_ears(tmp_path):
    encoder = _tiny_pointnet_config()
    locator_config = {
        "backbone": "pointnet2",
        "encoder_config": encoder,
        "head_channels": [16],
        "dropout": 0.0,
    }
    landmark_config = {
        "backbone": "pointnet2",
        "encoder_config": encoder,
        "four_heads": True,
        "head_channels": [16],
        "dropout": 0.0,
        "refinement_k": 0,
        "refinement_cap_normalized": 0.0,
    }
    from src.proposal_models import build_landmark_model, build_locator

    locator = build_locator(locator_config)
    landmark = build_landmark_model(landmark_config)
    for model in (locator, landmark):
        for parameter in model.parameters():
            parameter.data.zero_()
    bundle = {
        "schema_version": 2,
        "locator": {"model_config": locator_config, "state_dict": locator.state_dict()},
        "landmark": {"model_config": landmark_config, "state_dict": landmark.state_dict()},
        "broad_config": {
            "box": {"min": [-2.0, -2.0, -2.0], "max": [2.0, 2.0, 2.0]},
            "initial_center": [0.0, 0.0, 0.0],
            "input_scale": 2.0,
            "margin": 0.2,
        },
        "crop_calibration": {
            "primary": {"negative": [2.0, 2.0, 2.0], "positive": [2.0, 2.0, 2.0]},
            "backup": {"negative": [2.0, 2.0, 2.0], "positive": [2.0, 2.0, 2.0]},
            "fallback_thresholds": {"face_count_p01": 0, "surface_area_p01": 0.0},
            "local_scale": 2.0,
        },
        "coordinates": {
            "local_frame": "canonical_left_ear",
            "right_reflection": [1.0, -1.0, 1.0],
            "units": "millimetres",
            "local_scale": 2.0,
        },
        "sampling": {"locator_points": 64, "landmark_points": 64, "seed": 42},
        "postprocess": {"project_to_surface": False},
    }
    checkpoint = tmp_path / "v2.pt"
    torch.save(bundle, checkpoint)
    extractor = LandmarkExtractor(str(checkpoint), device="cpu")
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    first = extractor.extract(mesh)
    second = extractor.extract(mesh)
    for actual, repeated in zip(first, second):
        assert actual.shape == (85, 3)
        assert actual.dtype == np.float32
        assert np.isfinite(actual).all()
        assert np.array_equal(actual, repeated)


def test_incomplete_v2_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "broken.pt"
    torch.save({"schema_version": 2}, path)
    with pytest.raises(ValueError, match="Incomplete v2 pipeline checkpoint"):
        LandmarkExtractor(str(path), device="cpu")
