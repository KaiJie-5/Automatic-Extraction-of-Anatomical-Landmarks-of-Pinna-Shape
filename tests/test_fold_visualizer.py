import argparse
import csv
import hashlib
import json

import numpy as np
import pytest
import torch
import trimesh

from train_pipeline import command_evaluate_projection
from src.meshnet import meshnet_inputs_with_mesh
from src.pipeline_dataset import (
    EarLandmarkDataset,
    EarMeshLandmarkDataset,
    prepare_ear_geometry,
)
from src.proposal_models import build_fold_landmark_model
from visualize_point_importance import (
    _gallery_selection,
    _write_landmark_csv,
    build_arg_parser,
    deterministic_spatial_clusters,
    export_importance_artifacts,
    gradient_importance,
    load_fold_context,
    occlusion_importance,
    prepare_trace,
    validation_sample_seed,
)


def _write_landmarks(path, points):
    with path.open("w", encoding="utf-8") as handle:
        for index, point in enumerate(points):
            handle.write(f"{index},[{point[0]} {point[1]} {point[2]}]\n")


def _tiny_encoder_config():
    return {
        "input_channels": 6,
        "use_normals": True,
        "variant": "ssg",
        "ssg_npoints": [16, 4],
        "ssg_radii": [0.4, 0.8],
        "ssg_nsamples": [8, 8],
        "ssg_mlps": [[8, 8, 16], [16, 16, 32], [32, 64]],
    }


def _make_artifacts(tmp_path, *, stored_seed=False, hashes=False):
    mesh_dir = tmp_path / "mesh"
    landmarks_dir = tmp_path / "landmarks"
    mesh_dir.mkdir()
    landmarks_dir.mkdir()
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    angle = np.linspace(0.0, 2.0 * np.pi, 85, endpoint=False, dtype=np.float32)
    left = np.stack(
        [3.0 * np.cos(angle), np.full(85, 2.0), 4.0 * np.sin(angle)], axis=1
    ).astype(np.float32)
    right = left.copy()
    right[:, 1] *= -1.0
    for subject in ("TRAIN", "VAL"):
        mesh.export(mesh_dir / f"{subject}.ply")
        _write_landmarks(landmarks_dir / f"{subject}_left_ear_landmarks.csv", left)
        _write_landmarks(landmarks_dir / f"{subject}_right_ear_landmarks.csv", right)

    subject_ids = ["TRAIN", "VAL"]
    subject_checksum = hashlib.sha256("\n".join(subject_ids).encode("utf-8")).hexdigest()
    folds = {
        "subject_checksum": subject_checksum,
        "outer": [
            {
                "fold": 0,
                "train": ["TRAIN"],
                "validation": ["VAL"],
                "inner": [],
            }
        ],
    }
    folds_path = tmp_path / "folds.json"
    folds_path.write_text(json.dumps(folds), encoding="utf-8")
    calibration = {
        "schema_version": 1,
        "primary": {
            "negative": [20.0, 20.0, 20.0],
            "positive": [20.0, 20.0, 20.0],
            "complete_ear_coverage": 1.0,
        },
        "backup": {
            "negative": [24.0, 24.0, 24.0],
            "positive": [24.0, 24.0, 24.0],
            "complete_ear_coverage": 1.0,
            "expansion": 0.2,
        },
        "fallback_thresholds": {"face_count_p01": 0, "surface_area_p01": 0.0},
        "local_scale": 20.0,
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    centers = {
        f"{subject}:{ear}": [0.0, 2.0, 0.0]
        for subject in subject_ids
        for ear in ("left", "right")
    }
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {"coordinate_frame": "canonical_mm", "center_predictions": centers}
        ),
        encoding="utf-8",
    )
    model_config = {
        "backbone": "pointnet2",
        "encoder_config": _tiny_encoder_config(),
        "four_heads": True,
        "head_channels": [16],
        "dropout": 0.0,
        "refinement_k": 0,
        "refinement_cap_normalized": 0.0,
    }
    model = build_fold_landmark_model(model_config)
    data_config = {
        "outer_fold": 0,
        "train_ids": ["TRAIN"],
        "validation_ids": ["VAL"],
        "calibration": calibration,
        "num_points": 64,
    }
    if stored_seed:
        data_config["seed"] = 42
    if hashes:
        from visualize_point_importance import file_sha256

        data_config["artifact_checksums"] = {
            "folds_json_sha256": file_sha256(folds_path),
            "predictions_json_sha256": file_sha256(predictions_path),
            "calibration_json_sha256": file_sha256(calibration_path),
        }
    checkpoint_path = tmp_path / "best_landmarks.pt"
    torch.save(
        {
            "component_schema_version": 1,
            "component": "landmarks",
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "data_config": data_config,
        },
        checkpoint_path,
    )
    return {
        "mesh_dir": mesh_dir,
        "landmarks_dir": landmarks_dir,
        "folds": folds_path,
        "calibration": calibration_path,
        "predictions": predictions_path,
        "checkpoint": checkpoint_path,
        "calibration_data": calibration,
        "centers": centers,
    }


def _args(paths, *, run_seed=42):
    values = [
        "single",
        "--checkpoint-path", str(paths["checkpoint"]),
        "--predictions-json", str(paths["predictions"]),
        "--calibration-json", str(paths["calibration"]),
        "--folds-json", str(paths["folds"]),
        "--mesh-dir", str(paths["mesh_dir"]),
        "--landmarks-dir", str(paths["landmarks_dir"]),
        "--subject-id", "VAL",
        "--ear", "left",
        "--output-dir", str(paths["checkpoint"].parent / "output"),
        "--device", "cpu",
    ]
    if run_seed is not None:
        values.extend(["--run-seed", str(run_seed)])
    return build_arg_parser().parse_args(values)


def test_existing_checkpoint_requires_explicit_seed(tmp_path):
    paths = _make_artifacts(tmp_path)
    with pytest.raises(ValueError, match="provide --run-seed"):
        load_fold_context(_args(paths, run_seed=None))


def test_future_checkpoint_seed_and_hashes_are_verified(tmp_path):
    paths = _make_artifacts(tmp_path, stored_seed=True, hashes=True)
    context = load_fold_context(_args(paths, run_seed=None))
    assert context.run_seed == 42
    assert context.provenance_level == "sha256_verified"
    paths["predictions"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_fold_context(_args(paths, run_seed=None))


def test_wrong_calibration_and_training_subject_are_rejected(tmp_path):
    paths = _make_artifacts(tmp_path)
    changed = json.loads(paths["calibration"].read_text(encoding="utf-8"))
    changed["local_scale"] = 21.0
    paths["calibration"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exactly match"):
        load_fold_context(_args(paths))
    paths["calibration"].write_text(
        json.dumps(paths["calibration_data"]), encoding="utf-8"
    )
    context = load_fold_context(_args(paths))
    with pytest.raises(ValueError, match="not held out"):
        validation_sample_seed(context, "TRAIN", "left")


def test_wrong_prediction_frame_and_missing_centres_are_rejected(tmp_path):
    paths = _make_artifacts(tmp_path)
    predictions = json.loads(paths["predictions"].read_text(encoding="utf-8"))
    predictions["coordinate_frame"] = "official_mm"
    paths["predictions"].write_text(json.dumps(predictions), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical_mm"):
        load_fold_context(_args(paths))
    predictions["coordinate_frame"] = "canonical_mm"
    predictions["center_predictions"].pop("VAL:right")
    paths["predictions"].write_text(json.dumps(predictions), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage"):
        load_fold_context(_args(paths))


@pytest.mark.parametrize(("ear", "item"), (("left", 0), ("right", 1)))
def test_viewer_reproduces_validation_dataset_sample_and_prediction(tmp_path, ear, item):
    paths = _make_artifacts(tmp_path)
    context = load_fold_context(_args(paths))
    trace = prepare_trace(context, "VAL", ear, include_projection=False)
    validation = EarLandmarkDataset(
        str(paths["mesh_dir"]),
        str(paths["landmarks_dir"]),
        paths["centers"],
        paths["calibration_data"],
        subject_ids=["VAL"],
        num_points=64,
        seed=42 + 100_000,
        dynamic_sampling=False,
        augment=False,
    )
    expected = validation[item]
    assert trace.sample_seed == 42 + 100_000 + item * 1009
    assert np.array_equal(trace.input_features, expected["points"].numpy())
    assert np.array_equal(trace.target_local, expected["landmarks"].numpy())
    with torch.no_grad():
        direct = context.model(expected["points"].unsqueeze(0)).squeeze(0).numpy()
    assert np.array_equal(trace.final_local, direct)


def test_quantitative_projection_evaluation_covers_the_held_out_fold(tmp_path):
    paths = _make_artifacts(tmp_path, stored_seed=True, hashes=True)
    output = tmp_path / "projection_evaluation.json"
    args = argparse.Namespace(
        checkpoint_path=str(paths["checkpoint"]),
        predictions_json=str(paths["predictions"]),
        calibration_json=str(paths["calibration"]),
        folds_json=str(paths["folds"]),
        mesh_dir=str(paths["mesh_dir"]),
        landmarks_dir=str(paths["landmarks_dir"]),
        run_seed=None,
        output=str(output),
        device="cpu",
    )
    command_evaluate_projection(args)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["component"] == "fold_surface_projection_evaluation"
    assert report["outer_fold"] == 0
    assert report["run_seed"] == 42
    assert report["subject_count"] == 1
    assert report["ear_count"] == 2
    assert set(report["per_ear"]) == {"VAL:left", "VAL:right"}
    assert len(report["raw"]["per_landmark_md_mm"]) == 85
    assert len(report["projected"]["per_landmark_md_mm"]) == 85
    assert np.isfinite(report["raw"]["pooled_md_mm"])
    assert np.isfinite(report["projected"]["pooled_md_mm"])


def test_forward_with_details_preserves_point_and_mesh_model_outputs():
    point_config = {
        "backbone": "pointnet2",
        "encoder_config": _tiny_encoder_config(),
        "four_heads": False,
        "head_channels": [16],
        "dropout": 0.0,
        "refinement_k": 0,
        "refinement_cap_normalized": 0.0,
    }
    point_model = build_fold_landmark_model(point_config).eval()
    points = torch.randn(1, 64, 6)
    with torch.no_grad():
        details = point_model.forward_with_details(points)
        direct = point_model(points)
    assert torch.equal(details["coarse"], details["final"])
    assert torch.equal(details["final"], direct)

    mesh_model = build_fold_landmark_model(
        {
            "backbone": "meshnet",
            "target_faces": 16,
            "input_dim": 15,
            "width": 16,
            "four_heads": True,
        }
    ).eval()
    faces = torch.randn(1, 16, 15)
    neighbors = torch.zeros(1, 16, 3, dtype=torch.long)
    with torch.no_grad():
        mesh_details = mesh_model.forward_with_details(faces, neighbors)
        mesh_direct = mesh_model(faces, neighbors)
    assert mesh_direct.shape == (1, 85, 3)
    assert torch.equal(mesh_details["coarse"], mesh_details["final"])
    assert torch.equal(mesh_details["final"], mesh_direct)


def test_meshnet_view_preparation_matches_validation_dataset(tmp_path):
    paths = _make_artifacts(tmp_path)
    dataset = EarMeshLandmarkDataset(
        str(paths["mesh_dir"]),
        str(paths["landmarks_dir"]),
        paths["centers"],
        paths["calibration_data"],
        subject_ids=["VAL"],
        num_points=64,
        seed=42 + 100_000,
        dynamic_sampling=False,
        augment=False,
        target_faces=64,
    )
    expected = dataset[0]
    mesh, left, _ = dataset.base[dataset.samples[0][0]]
    prepared = prepare_ear_geometry(
        mesh,
        left,
        "left",
        paths["centers"]["VAL:left"],
        paths["calibration_data"],
        1,
        42 + 100_000,
    )
    features, neighbors, simplified = meshnet_inputs_with_mesh(
        prepared.crop_mesh, 64, "left", prepared.transform
    )
    assert len(simplified.faces) == 64
    assert np.array_equal(features, expected["face_features"].numpy())
    assert np.array_equal(neighbors, expected["neighbors"].numpy())


def test_refinement_and_pointnext_detailed_outputs():
    refinement_config = {
        "backbone": "pointnet2",
        "encoder_config": _tiny_encoder_config(),
        "four_heads": True,
        "head_channels": [16],
        "dropout": 0.0,
        "refinement_k": 32,
        "refinement_cap_normalized": 0.25,
    }
    refinement = build_fold_landmark_model(refinement_config).eval()
    points = torch.randn(1, 64, 6)
    with torch.no_grad():
        details = refinement.forward_with_details(points)
        direct = refinement(points)
    assert details["coarse"].shape == (1, 85, 3)
    assert torch.equal(details["final"], direct)

    pointnext = build_fold_landmark_model(
        {
            "backbone": "pointnext",
            "encoder_config": {
                "input_channels": 6,
                "width": 8,
                "strides": [1, 2, 2, 2, 2],
                "blocks": [1, 1, 1, 1, 1],
                "radius": 0.1,
                "radius_scaling": 2.0,
                "nsample": 8,
                "expansion": 2,
            },
            "four_heads": False,
            "head_channels": [16],
            "dropout": 0.0,
            "refinement_k": 0,
            "refinement_cap_normalized": 0.0,
        }
    ).eval()
    with torch.no_grad():
        pointnext_details = pointnext.forward_with_details(points)
        pointnext_direct = pointnext(points)
    assert pointnext_direct.shape == (1, 85, 3)
    assert torch.equal(pointnext_details["final"], pointnext_direct)


def test_legacy_and_v2_checkpoints_are_rejected(tmp_path):
    paths = _make_artifacts(tmp_path)
    torch.save({"schema_version": 2}, paths["checkpoint"])
    with pytest.raises(ValueError, match="final v2"):
        load_fold_context(_args(paths))
    torch.save({"weight": torch.zeros(1)}, paths["checkpoint"])
    with pytest.raises(ValueError, match="component_schema_version"):
        load_fold_context(_args(paths))


def test_spatial_clusters_are_deterministic_and_cover_every_input():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(257, 3)).astype(np.float32)
    first = deterministic_spatial_clusters(points, target_size=32)
    second = deterministic_spatial_clusters(points, target_size=32)
    assert np.array_equal(first, second)
    assert first.shape == (257,)
    assert np.all(first >= 0)
    assert len(np.unique(first)) == int(np.ceil(257 / 32))


def test_gradient_and_occlusion_importance_are_finite_and_aligned(tmp_path):
    paths = _make_artifacts(tmp_path)
    context = load_fold_context(_args(paths))
    trace = prepare_trace(context, "VAL", "left", include_projection=False)
    gradients = gradient_importance(context, trace)
    occlusion = occlusion_importance(
        context, trace, cluster_size=16, batch_size=2
    )
    for values in (
        gradients["gradient_xyz"],
        gradients["gradient_normals"],
        occlusion["occlusion_md"],
        occlusion["occlusion_displacement"],
    ):
        assert values.shape == (64,)
        assert np.isfinite(values).all()
    assert occlusion["cluster_labels"].shape == (64,)


def test_importance_artifacts_and_landmark_csv_are_loadable(tmp_path):
    paths = _make_artifacts(tmp_path)
    context = load_fold_context(_args(paths))
    trace = prepare_trace(context, "VAL", "left", include_projection=False)
    base = tmp_path / "importance_gradient_xyz"
    export_importance_artifacts(
        base,
        context,
        trace,
        np.linspace(0.0, 1.0, len(trace.input_features), dtype=np.float32),
        signed=False,
    )
    assert base.with_suffix(".png").stat().st_size > 0
    assert base.with_suffix(".glb").stat().st_size > 0
    assert base.with_suffix(".ply").stat().st_size > 0
    scene = trimesh.load(base.with_suffix(".glb"), force="scene", process=False)
    assert len(scene.geometry) > 0
    csv_path = tmp_path / "landmarks.csv"
    _write_landmark_csv(csv_path, trace)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 85
    assert [int(row["index"]) for row in rows] == list(range(85))


def test_gallery_selection_uses_raw_ear_md_and_stable_ties():
    records = [
        {"subject_id": "B", "ear": "left", "raw_md_mm": 3.0},
        {"subject_id": "A", "ear": "right", "raw_md_mm": 1.0},
        {"subject_id": "A", "ear": "left", "raw_md_mm": 2.0},
        {"subject_id": "B", "ear": "right", "raw_md_mm": 4.0},
    ]
    selected = _gallery_selection(records)
    assert selected["best"]["raw_md_mm"] == 1.0
    assert selected["median"]["subject_id"] == "A"
    assert selected["median"]["ear"] == "left"
    assert selected["worst"]["raw_md_mm"] == 4.0
