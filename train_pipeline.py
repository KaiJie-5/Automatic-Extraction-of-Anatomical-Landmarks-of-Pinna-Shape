"""Unified proposal-aligned pipeline CLI.

Run ``python train_pipeline.py <stage> --help`` for stage-specific arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from src.audit import audit_dataset
from src.calibration import (
    EarCalibrationRecord,
    boxes_for_prediction,
    calibrate_directional_crops,
    evaluate_calibration,
    fit_broad_box,
    fit_broad_box_with_margin,
)
from src.canonical import WorldCropBox, canonicalize_xyz, decanonicalize_xyz
from src.dataset import Dataset
from src.estimator import LandmarkExtractor
from src.geometry import clip_mesh_to_box, crop_geometry_stats, sample_canonical_crop
from src.meshnet import run_meshnet_gate
from src.metrics import compute_mean_landmark_distance
from src.losses import candidate_is_promoted
from src.pipeline_dataset import (
    EarLandmarkDataset,
    EarLocatorDataset,
    EarMeshLandmarkDataset,
    prediction_key,
)
from src.pointnet2_model import default_model_config
from src.pointnext_model import default_pointnext_config
from src.proposal_models import build_fold_landmark_model, build_locator
from src.splits import folds_from_audit, save_folds
from src.training import (
    predict_locator,
    resolve_device,
    seed_everything,
    train_landmarks,
    train_locator,
)


PROJECT_ROOT = Path(
    "/iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape"
)
DATA_ROOT = PROJECT_ROOT / "data"


def read_json(path: str):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def pointnet_encoder_config() -> dict:
    config = default_model_config()
    for key in ("num_landmarks", "head_channels", "dropout"):
        config.pop(key, None)
    return config


def locator_model_config(backbone: str = "pointnet2") -> dict:
    encoder = pointnet_encoder_config() if backbone == "pointnet2" else default_pointnext_config()
    return {
        "backbone": backbone,
        "encoder_config": encoder,
        "head_channels": [256, 128],
        "dropout": 0.0,
    }


def landmark_model_config(args, local_scale: float) -> dict:
    if args.backbone == "meshnet":
        gate = read_json(args.meshnet_gate_json) if args.meshnet_gate_json else None
        if not gate or not gate.get("passed") or not gate.get("target_faces"):
            raise ValueError("MeshNet requires a passing --meshnet-gate-json report")
        return {
            "backbone": "meshnet",
            "target_faces": int(gate["target_faces"]),
            "input_dim": 15,
            "width": 128,
            "four_heads": bool(args.four_heads),
        }
    encoder = pointnet_encoder_config() if args.backbone == "pointnet2" else default_pointnext_config()
    return {
        "backbone": args.backbone,
        "encoder_config": encoder,
        "four_heads": bool(args.four_heads),
        "head_channels": [512, 256],
        "dropout": 0.0,
        "refinement_k": int(args.refinement_k),
        "refinement_cap_normalized": 5.0 / local_scale if args.refinement_k else 0.0,
    }


def make_landmark_model(config: Mapping[str, object]):
    return build_fold_landmark_model(config)


def make_landmark_dataset(args, predictions, calibration, subject_ids, seed, training):
    dense_points = 32768 if args.surface_weight else 0
    common = dict(
        mesh_dir=args.mesh_dir,
        landmarks_dir=args.landmarks_dir,
        center_predictions=predictions,
        calibration=calibration,
        subject_ids=subject_ids,
        num_points=args.num_points,
        dense_surface_points=dense_points,
        seed=seed,
        dynamic_sampling=training,
        augment=args.augment if training else False,
    )
    if args.backbone == "meshnet":
        gate = read_json(args.meshnet_gate_json)
        return EarMeshLandmarkDataset(
            **common, target_faces=int(gate["target_faces"])
        )
    return EarLandmarkDataset(**common)


def subject_ears(dataset: Dataset, subject_ids: Sequence[str]):
    index = {dataset.get_identifier(i): i for i in range(len(dataset))}
    result = []
    for subject_id in subject_ids:
        _, left, right = dataset[index[subject_id]]
        result.extend([canonicalize_xyz(left, "left"), canonicalize_xyz(right, "right")])
    return result


def validate_fold_dataset(dataset: Dataset, folds: Mapping[str, object]) -> None:
    subject_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
    checksum = hashlib.sha256("\n".join(sorted(subject_ids)).encode("utf-8")).hexdigest()
    if checksum != folds.get("subject_checksum"):
        raise ValueError(
            "dataset subject IDs do not match the audited fold manifest; rerun audit and make-folds"
        )


def select_outer_fold(folds: Mapping[str, object], index: int):
    for record in folds["outer"]:
        if int(record["fold"]) == index:
            return record
    raise ValueError(f"outer fold {index} does not exist")


def load_best_component(path: Path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return checkpoint


def command_audit(args):
    report = audit_dataset(
        args.mesh_dir,
        args.landmarks_dir,
        args.output_root,
        fail_on_fatal=True,
        expected_subjects=args.expected_subjects,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Audit report: {report['output_dir']}")


def command_make_folds(args):
    if args.audit_json:
        report = read_json(args.audit_json)
    else:
        report = audit_dataset(
            args.mesh_dir,
            args.landmarks_dir,
            fail_on_fatal=True,
            expected_subjects=args.expected_subjects,
        )
    if int(report["summary"].get("fatal_issues", 0)):
        raise ValueError("cannot create folds from an audit report containing fatal issues")
    if int(report["summary"]["complete_subjects"]) != int(args.expected_subjects):
        raise ValueError(
            f"expected {args.expected_subjects} audited subjects; "
            f"found {report['summary']['complete_subjects']}"
        )
    folds = folds_from_audit(report, seed=args.seed)
    save_folds(folds, args.output)
    print(f"Wrote {args.output} with outer sizes {[len(item['validation']) for item in folds['outer']]}")


def _fit_locator_fold(args, outer_index: int):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    folds = read_json(args.folds_json)
    outer = select_outer_fold(folds, outer_index)
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    validate_fold_dataset(dataset, folds)
    output = Path(args.output_dir) / f"outer_{outer_index}"
    output.mkdir(parents=True, exist_ok=True)
    model_config = locator_model_config("pointnet2")
    inner_predictions = {}
    margins = []
    inner_metrics = []

    for inner in outer["inner"]:
        inner_index = int(inner["fold"])
        broad = fit_broad_box(
            subject_ears(dataset, inner["train"]),
            subject_ears(dataset, inner["validation"]),
        )
        margins.append(float(broad["margin"]))
        train_data = EarLocatorDataset(
            args.mesh_dir, args.landmarks_dir, broad, inner["train"], args.num_points,
            args.seed + outer_index * 100 + inner_index, True,
        )
        validation_data = EarLocatorDataset(
            args.mesh_dir, args.landmarks_dir, broad, inner["validation"], args.num_points,
            args.seed + 100_000 + outer_index * 100 + inner_index, False,
        )
        model = build_locator(model_config)
        run_dir = output / f"inner_{inner_index}"
        metrics = train_locator(
            model, train_data, validation_data, str(run_dir), model_config,
            {"broad_config": broad, "train_ids": inner["train"], "validation_ids": inner["validation"]},
            device, args.epochs, args.batch_size, 32, args.workers, 1e-3, 1e-4,
            args.patience, args.amp, not args.no_resume,
        )
        load_best_component(run_dir / "best_locator.pt", model, device)
        predictions = predict_locator(model, validation_data, broad, device, args.workers)
        overlap = set(inner_predictions) & set(predictions)
        if overlap:
            raise RuntimeError(f"duplicate inner OOF predictions: {sorted(overlap)[:3]}")
        inner_predictions.update(predictions)
        write_json(run_dir / "broad_config.json", broad)
        inner_metrics.append({"fold": inner_index, **metrics, "margin": broad["margin"]})

    selected_margin = max(margins)
    outer_broad = fit_broad_box_with_margin(subject_ears(dataset, outer["train"]), selected_margin)
    train_data = EarLocatorDataset(
        args.mesh_dir, args.landmarks_dir, outer_broad, outer["train"], args.num_points,
        args.seed + outer_index, True,
    )
    validation_data = EarLocatorDataset(
        args.mesh_dir, args.landmarks_dir, outer_broad, outer["validation"], args.num_points,
        args.seed + 200_000 + outer_index, False,
    )
    model = build_locator(model_config)
    outer_run = output / "outer_model"
    outer_metrics = train_locator(
        model, train_data, validation_data, str(outer_run), model_config,
        {"broad_config": outer_broad, "train_ids": outer["train"], "validation_ids": outer["validation"]},
        device, args.epochs, args.batch_size, 32, args.workers, 1e-3, 1e-4,
        args.patience, args.amp, not args.no_resume,
    )
    load_best_component(outer_run / "best_locator.pt", model, device)
    outer_predictions = predict_locator(model, validation_data, outer_broad, device, args.workers)
    write_json(output / "broad_config.json", outer_broad)
    write_json(
        output / "oof_predictions.json",
        {
            "schema_version": 1,
            "coordinate_frame": "canonical_mm",
            "outer_fold": outer_index,
            "inner_oof": inner_predictions,
            "outer_oof": outer_predictions,
        },
    )
    manifest = {
        "outer_fold": outer_index,
        "inner": inner_metrics,
        "outer": outer_metrics,
        "selected_broad_margin": selected_margin,
        "inner_prediction_count": len(inner_predictions),
        "outer_prediction_count": len(outer_predictions),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def command_fit_locator(args):
    if args.outer_fold == "all":
        for outer_index in range(5):
            _fit_locator_fold(args, outer_index)
    else:
        _fit_locator_fold(args, int(args.outer_fold))


def collect_oof_predictions(root: str, outer_fold: int | None):
    files = sorted(Path(root).glob("outer_*/oof_predictions.json"))
    if not files:
        raise FileNotFoundError(f"no outer_*/oof_predictions.json files under {root}")
    predictions = {}
    for path in files:
        data = read_json(str(path))
        if outer_fold is not None and int(data["outer_fold"]) != outer_fold:
            continue
        source = data["inner_oof"] if outer_fold is not None else data["outer_oof"]
        overlap = set(predictions) & set(source)
        if overlap:
            raise ValueError(f"duplicate OOF prediction keys: {sorted(overlap)[:3]}")
        predictions.update(source)
    return predictions


def command_calibrate(args):
    outer_fold = None if args.outer_fold == "final" else int(args.outer_fold)
    predictions = collect_oof_predictions(args.locator_run_root, outer_fold)
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    records = _calibration_records(dataset, predictions)
    index = {dataset.get_identifier(i): i for i in range(len(dataset))}
    preliminary = calibrate_directional_crops(records)
    geometry = []
    for record in records:
        mesh, _, _ = dataset[index[record.subject_id]]
        canonical_prediction = canonicalize_xyz(record.predicted_center, record.ear)
        primary, _ = boxes_for_prediction(canonical_prediction, preliminary)
        crop = clip_mesh_to_box(mesh, primary.for_ear(record.ear))
        geometry.append(crop_geometry_stats(crop))
    calibration = calibrate_directional_crops(records, geometry_stats=geometry)
    prediction_output = dict(predictions)
    if outer_fold is not None:
        matching = read_json(
            str(Path(args.locator_run_root) / f"outer_{outer_fold}" / "oof_predictions.json")
        )
        overlap = set(prediction_output) & set(matching["outer_oof"])
        if overlap:
            raise ValueError(f"train and validation prediction keys overlap: {sorted(overlap)[:3]}")
        prediction_output.update(matching["outer_oof"])
    output = Path(args.output)
    write_json(output, calibration)
    write_json(
        output.with_name(output.stem + "_predictions.json"),
        {"coordinate_frame": "canonical_mm", "center_predictions": prediction_output},
    )
    print(json.dumps(calibration, indent=2, sort_keys=True))


def _calibration_records(dataset: Dataset, predictions: Mapping[str, Sequence[float]]):
    index = {dataset.get_identifier(i): i for i in range(len(dataset))}
    records = []
    for key, canonical_prediction in predictions.items():
        subject_id, ear = key.split(":")
        if subject_id not in index:
            raise ValueError(f"prediction references unknown subject {subject_id}")
        if ear not in {"left", "right"}:
            raise ValueError(f"prediction key has invalid ear: {key}")
        _, left, right = dataset[index[subject_id]]
        landmarks = left if ear == "left" else right
        original_prediction = decanonicalize_xyz(np.asarray(canonical_prediction, dtype=np.float32), ear)
        records.append(EarCalibrationRecord(subject_id, ear, landmarks, original_prediction))
    return records


def _require_prediction_keys(
    predictions: Mapping[str, Sequence[float]], subject_ids: Sequence[str]
) -> None:
    expected = {
        f"{subject_id}:{ear}"
        for subject_id in subject_ids
        for ear in ("left", "right")
    }
    actual = set(predictions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "calibration prediction coverage is incomplete: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def command_validate_calibration(args):
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    if args.outer_fold == "all":
        if not args.folds_json:
            raise ValueError("--folds-json is required when validating all outer folds")
        if "{fold}" not in args.calibration_json or "{fold}" not in args.predictions_json:
            raise ValueError(
                "--outer-fold all requires {fold} in both calibration and prediction paths"
            )
        folds = read_json(args.folds_json)
        validate_fold_dataset(dataset, folds)
        fold_reports = []
        for outer_index in range(5):
            calibration = read_json(args.calibration_json.format(fold=outer_index))
            predictions = _load_prediction_map(
                args.predictions_json.format(fold=outer_index)
            )
            validation_subjects = set(
                select_outer_fold(folds, outer_index)["validation"]
            )
            selected = {
                key: value
                for key, value in predictions.items()
                if key.split(":", 1)[0] in validation_subjects
            }
            _require_prediction_keys(selected, sorted(validation_subjects))
            fold_report = evaluate_calibration(
                _calibration_records(dataset, selected), calibration
            )
            fold_report["outer_fold"] = outer_index
            fold_reports.append(fold_report)
        ear_count = sum(item["ear_count"] for item in fold_reports)
        primary_complete = sum(
            item["primary"]["complete_count"] for item in fold_reports
        )
        backup_complete = sum(
            item["backup"]["complete_count"] for item in fold_reports
        )
        report = {
            "scope": "pooled held-out predictions from all five outer folds",
            "ear_count": ear_count,
            "primary": {
                "complete_count": primary_complete,
                "coverage": primary_complete / ear_count,
                "misses": [
                    miss
                    for item in fold_reports
                    for miss in item["primary"]["misses"]
                ],
            },
            "backup": {
                "complete_count": backup_complete,
                "coverage": backup_complete / ear_count,
                "misses": [
                    miss
                    for item in fold_reports
                    for miss in item["backup"]["misses"]
                ],
                "is_primary_superset": all(
                    item["backup"]["is_primary_superset"] for item in fold_reports
                ),
            },
            "folds": fold_reports,
        }
    else:
        calibration = read_json(args.calibration_json)
        predictions = _load_prediction_map(args.predictions_json)
        if args.outer_fold == "final":
            selected = predictions
            scope = "all full-dataset out-of-fold predictions"
            subject_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
            _require_prediction_keys(selected, subject_ids)
        else:
            if not args.folds_json:
                raise ValueError("--folds-json is required when validating an outer fold")
            outer_index = int(args.outer_fold)
            folds = read_json(args.folds_json)
            validate_fold_dataset(dataset, folds)
            validation_subjects = set(select_outer_fold(folds, outer_index)["validation"])
            selected = {
                key: value
                for key, value in predictions.items()
                if key.split(":", 1)[0] in validation_subjects
            }
            _require_prediction_keys(selected, sorted(validation_subjects))
            scope = f"outer fold {outer_index} held-out subjects"
        report = evaluate_calibration(
            _calibration_records(dataset, selected), calibration
        )
        report["scope"] = scope

    report["requirements"] = {
        "primary_coverage_at_least": 0.99,
        "backup_coverage": 1.0,
        "backup_is_primary_superset": True,
    }
    report["passed"] = bool(
        report["primary"]["coverage"] >= 0.99
        and report["backup"]["coverage"] == 1.0
        and report["backup"]["is_primary_superset"]
    )
    if args.output:
        write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def _load_prediction_map(path: str):
    data = read_json(path)
    return data.get("center_predictions", data)


def command_fit_landmarks(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    folds = read_json(args.folds_json)
    outer = select_outer_fold(folds, args.outer_fold)
    calibration = read_json(args.calibration_json)
    predictions = _load_prediction_map(args.predictions_json)
    validate_fold_dataset(Dataset(args.mesh_dir, args.landmarks_dir), folds)
    dense_points = 32768 if args.surface_weight else 0
    train_data = make_landmark_dataset(
        args, predictions, calibration, outer["train"], args.seed, True
    )
    validation_data = make_landmark_dataset(
        args, predictions, calibration, outer["validation"], args.seed + 100_000, False
    )
    model_config = landmark_model_config(args, float(calibration["local_scale"]))
    model = make_landmark_model(model_config)
    loss_weights = {
        "anchor": args.anchor_weight,
        "spacing": args.spacing_weight,
        "surface": args.surface_weight,
    }
    data_config = {
        "outer_fold": args.outer_fold,
        "train_ids": outer["train"],
        "validation_ids": outer["validation"],
        "calibration": calibration,
        "num_points": args.num_points,
        "seed": args.seed,
        "artifact_checksums": {
            "folds_json_sha256": file_sha256(args.folds_json),
            "predictions_json_sha256": file_sha256(args.predictions_json),
            "calibration_json_sha256": file_sha256(args.calibration_json),
        },
        "dense_surface_points": dense_points,
        "augmentation": args.augment,
        "loss_weights": loss_weights,
    }
    metrics = train_landmarks(
        model, train_data, validation_data, args.output_dir, model_config, data_config,
        loss_weights, device, args.epochs, args.batch_size, 32, args.workers, 1e-3,
        1e-4, args.patience, args.amp, not args.no_resume,
    )
    write_json(Path(args.output_dir) / "run_manifest.json", {"model_config": model_config, "data_config": data_config, "metrics": metrics})
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _final_broad_config(locator_root: str, dataset: Dataset):
    manifests = [read_json(str(path)) for path in sorted(Path(locator_root).glob("outer_*/manifest.json"))]
    if len(manifests) != 5:
        raise ValueError("final training requires all five completed outer locator manifests")
    margin = max(float(item["selected_broad_margin"]) for item in manifests)
    all_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
    return fit_broad_box_with_margin(subject_ears(dataset, all_ids), margin)


def _bundle_v2(locator_checkpoint, landmark_checkpoint, broad, calibration, args, subject_ids):
    subject_checksum = hashlib.sha256("\n".join(sorted(subject_ids)).encode("utf-8")).hexdigest()
    return {
        "schema_version": 2,
        "pipeline": "proposal_coarse_to_fine",
        "locator": {
            "model_config": locator_checkpoint["model_config"],
            "state_dict": locator_checkpoint["model_state_dict"],
        },
        "landmark": {
            "model_config": landmark_checkpoint["model_config"],
            "state_dict": landmark_checkpoint["model_state_dict"],
        },
        "broad_config": broad,
        "crop_calibration": calibration,
        "coordinates": {
            "local_frame": "canonical_left_ear",
            "right_reflection": [1.0, -1.0, 1.0],
            "units": "millimetres",
            "local_scale": float(calibration["local_scale"]),
        },
        "sampling": {"locator_points": args.num_points, "landmark_points": args.num_points, "seed": args.seed},
        "postprocess": {"project_to_surface": bool(args.project_to_surface)},
        "training": {
            "subject_ids": subject_ids,
            "subject_count": len(subject_ids),
            "subject_checksum": subject_checksum,
            "locator_metrics": locator_checkpoint.get("metrics", {}),
            "landmark_metrics": landmark_checkpoint.get("metrics", {}),
        },
    }


def command_fit_final(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    subject_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
    predictions = collect_oof_predictions(args.locator_run_root, None)
    expected = {prediction_key(subject_id, ear) for subject_id in subject_ids for ear in ("left", "right")}
    if set(predictions) != expected:
        raise ValueError(f"final training needs {len(expected)} OOF ear centres; found {len(predictions)}")
    calibration = read_json(args.calibration_json)
    broad = _final_broad_config(args.locator_run_root, dataset)
    output = Path(args.output_dir)

    locator_config = locator_model_config("pointnet2")
    locator_model = build_locator(locator_config)
    locator_data = EarLocatorDataset(
        args.mesh_dir, args.landmarks_dir, broad, subject_ids, args.num_points, args.seed, True
    )
    train_locator(
        locator_model, locator_data, None, str(output / "locator"), locator_config,
        {"broad_config": broad, "train_ids": subject_ids}, device, args.locator_epochs,
        args.batch_size, 32, args.workers, 1e-3, 1e-4, 30, args.amp, not args.no_resume,
    )
    locator_checkpoint = torch.load(output / "locator" / "best_locator.pt", map_location="cpu")

    landmark_config = landmark_model_config(args, float(calibration["local_scale"]))
    landmark_model = make_landmark_model(landmark_config)
    dense = 32768 if args.surface_weight else 0
    landmark_data = make_landmark_dataset(
        args, predictions, calibration, subject_ids, args.seed, True
    )
    weights = {"anchor": args.anchor_weight, "spacing": args.spacing_weight, "surface": args.surface_weight}
    train_landmarks(
        landmark_model, landmark_data, None, str(output / "landmarks"), landmark_config,
        {"calibration": calibration, "train_ids": subject_ids, "loss_weights": weights},
        weights, device, args.landmark_epochs, args.batch_size, 32, args.workers,
        1e-3, 1e-4, 30, args.amp, not args.no_resume,
    )
    landmark_checkpoint = torch.load(output / "landmarks" / "best_landmarks.pt", map_location="cpu")
    bundle = _bundle_v2(locator_checkpoint, landmark_checkpoint, broad, calibration, args, subject_ids)
    checkpoint_path = output / "final_pipeline.pt"
    torch.save(bundle, checkpoint_path)
    write_json(output / "final_manifest.json", {key: value for key, value in bundle.items() if key not in {"locator", "landmark"}})
    print(f"Wrote v2 pipeline checkpoint: {checkpoint_path}")


def command_evaluate(args):
    extractor = LandmarkExtractor(args.checkpoint, seed=args.seed, device=args.device)
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    scores = {}
    timings = []
    landmark_errors = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for index in range(len(dataset)):
        subject_id = dataset.get_identifier(index)
        mesh, left_target, right_target = dataset[index]
        start = time.perf_counter()
        left, right = extractor.extract(mesh)
        if extractor.device.type == "cuda":
            torch.cuda.synchronize(extractor.device)
        timings.append((time.perf_counter() - start) * 1000.0)
        if index == 0:
            repeated_left, repeated_right = extractor.extract(mesh)
            if not np.array_equal(left, repeated_left) or not np.array_equal(right, repeated_right):
                raise RuntimeError("inference is not deterministic for a fixed mesh and seed")
        scores[subject_id] = {
            "left": compute_mean_landmark_distance(left, left_target),
            "right": compute_mean_landmark_distance(right, right_target),
        }
        landmark_errors.extend(
            [np.linalg.norm(left - left_target, axis=1), np.linalg.norm(right - right_target, axis=1)]
        )
    pooled = float(np.mean([value for item in scores.values() for value in item.values()]))
    peak_memory = (
        float(torch.cuda.max_memory_allocated(extractor.device) / 1024**2)
        if extractor.device.type == "cuda"
        else 0.0
    )
    write_json(
        Path(args.output),
        {
            "pooled_md_mm": pooled,
            "per_landmark_md_mm": np.mean(np.stack(landmark_errors), axis=0).tolist(),
            "runtime": {
                "device": str(extractor.device),
                "median_ms_per_mesh": float(np.median(timings)),
                "mean_ms_per_mesh": float(np.mean(timings)),
                "peak_gpu_memory_mb": peak_memory,
            },
            "subjects": scores,
        },
    )
    print(f"Pooled MD: {pooled:.6f} mm")


def command_promote(args):
    baseline = read_json(args.baseline)
    candidate = read_json(args.candidate)
    promoted = candidate_is_promoted(
        baseline["fold_md_mm"],
        candidate["fold_md_mm"],
        float(baseline["runtime_ms"]),
        float(candidate["runtime_ms"]),
    )
    result = {
        "promoted": promoted,
        "baseline": args.baseline,
        "candidate": args.candidate,
        "improved_folds": sum(
            c < b for b, c in zip(baseline["fold_md_mm"], candidate["fold_md_mm"])
        ),
        "baseline_pooled_md_mm": float(np.mean(baseline["fold_md_mm"])),
        "candidate_pooled_md_mm": float(np.mean(candidate["fold_md_mm"])),
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_meshnet_gate(args):
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    predictions = _load_prediction_map(args.predictions_json)
    calibration = read_json(args.calibration_json)

    def crops():
        for index in range(len(dataset)):
            subject_id = dataset.get_identifier(index)
            mesh, _, _ = dataset[index]
            for ear in ("left", "right"):
                center = np.asarray(predictions[prediction_key(subject_id, ear)], dtype=np.float32)
                primary, backup = boxes_for_prediction(center, calibration)
                _, selected_crop, _ = sample_canonical_crop(
                    mesh,
                    primary,
                    ear,
                    num_points=1,
                    seed=args.seed + index * 2 + (0 if ear == "left" else 1),
                    fallback_box=backup,
                    thresholds=calibration.get("fallback_thresholds"),
                )
                yield prediction_key(subject_id, ear), selected_crop

    result = run_meshnet_gate(crops())
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_package(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 2:
        raise ValueError("package requires a complete schema-version-2 pipeline checkpoint")
    required = {"locator", "landmark", "broad_config", "crop_calibration", "coordinates", "sampling", "postprocess"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"v2 checkpoint is incomplete: {missing}")
    root = Path(__file__).resolve().parent
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [
            root / "src",
            root / "configs",
            root / "__init__.py",
            root / "requirements.txt",
            root / "README.md",
            root / "TECHNICAL_METHOD.md",
            root / "RESEARCH_BASIS.md",
            root / "THIRD_PARTY_NOTICES.md",
        ]:
            if path.is_dir():
                for child in path.rglob("*"):
                    if not child.is_file() or child.suffix not in {".py", ".json"}:
                        continue
                    archive.write(child, child.relative_to(root))
            elif path.exists():
                archive.write(path, path.relative_to(root))
        archive.write(args.checkpoint, "checkpoints/final_pipeline.pt")
        archive.write(__file__, Path(__file__).name)
    print(f"Wrote submission package: {output}")


def add_data_arguments(parser):
    parser.add_argument("--mesh-dir", default=str(DATA_ROOT / "mesh"))
    parser.add_argument("--landmarks-dir", default=str(DATA_ROOT / "landmarks"))


def add_runtime_arguments(parser):
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=0, help="0 probes 32,16,8,4,2,1")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-resume", action="store_true")


def add_landmark_model_arguments(parser):
    parser.add_argument("--backbone", choices=("pointnet2", "pointnext", "meshnet"), default="pointnet2")
    parser.add_argument("--meshnet-gate-json")
    parser.add_argument("--four-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refinement-k", type=int, choices=(0, 32, 64), default=0)
    parser.add_argument("--anchor-weight", type=float, choices=(0.0, 0.01, 0.05, 0.1), default=0.0)
    parser.add_argument("--spacing-weight", type=float, choices=(0.0, 0.01, 0.05, 0.1), default=0.0)
    parser.add_argument("--surface-weight", type=float, choices=(0.0, 0.01, 0.05, 0.1), default=0.0)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    add_data_arguments(audit)
    audit.add_argument(
        "--output-root", default=str(PROJECT_ROOT / "dataset_analysis_outputs")
    )
    audit.add_argument("--expected-subjects", type=int, default=201)
    audit.set_defaults(function=command_audit)

    folds = subparsers.add_parser("make-folds")
    add_data_arguments(folds)
    folds.add_argument("--audit-json")
    folds.add_argument("--output", default="artifacts/folds.json")
    folds.add_argument("--seed", type=int, default=42)
    folds.add_argument("--expected-subjects", type=int, default=201)
    folds.set_defaults(function=command_make_folds)

    locator = subparsers.add_parser("fit-locator")
    add_data_arguments(locator)
    add_runtime_arguments(locator)
    locator.add_argument("--folds-json", required=True)
    locator.add_argument("--outer-fold", default="all", help="0-4 or all")
    locator.add_argument("--output-dir", default="runs/locator_cv")
    locator.set_defaults(function=command_fit_locator)

    calibrate = subparsers.add_parser("calibrate")
    add_data_arguments(calibrate)
    calibrate.add_argument("--locator-run-root", required=True)
    calibrate.add_argument("--outer-fold", default="final", help="0-4 for CV or final")
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(function=command_calibrate)

    validate_calibration = subparsers.add_parser("validate-calibration")
    add_data_arguments(validate_calibration)
    validate_calibration.add_argument("--calibration-json", required=True)
    validate_calibration.add_argument("--predictions-json", required=True)
    validate_calibration.add_argument("--folds-json")
    validate_calibration.add_argument(
        "--outer-fold",
        default="final",
        help="0-4, all for pooled held-out validation, or final",
    )
    validate_calibration.add_argument("--output")
    validate_calibration.set_defaults(function=command_validate_calibration)

    landmarks = subparsers.add_parser("fit-landmarks")
    add_data_arguments(landmarks)
    add_runtime_arguments(landmarks)
    add_landmark_model_arguments(landmarks)
    landmarks.add_argument("--folds-json", required=True)
    landmarks.add_argument("--outer-fold", type=int, required=True)
    landmarks.add_argument("--predictions-json", required=True)
    landmarks.add_argument("--calibration-json", required=True)
    landmarks.add_argument("--output-dir", required=True)
    landmarks.set_defaults(function=command_fit_landmarks)

    final = subparsers.add_parser("fit-final")
    add_data_arguments(final)
    add_runtime_arguments(final)
    add_landmark_model_arguments(final)
    final.add_argument("--locator-run-root", required=True)
    final.add_argument("--calibration-json", required=True)
    final.add_argument("--locator-epochs", type=int, required=True)
    final.add_argument("--landmark-epochs", type=int, required=True)
    final.add_argument("--project-to-surface", action=argparse.BooleanOptionalAction, default=False)
    final.add_argument("--output-dir", default="checkpoints")
    final.set_defaults(function=command_fit_final)

    evaluate = subparsers.add_parser("evaluate")
    add_data_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(function=command_evaluate)

    gate = subparsers.add_parser("meshnet-gate")
    add_data_arguments(gate)
    gate.add_argument("--predictions-json", required=True)
    gate.add_argument("--calibration-json", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--seed", type=int, default=42)
    gate.set_defaults(function=command_meshnet_gate)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--baseline", required=True)
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--output", required=True)
    promote.set_defaults(function=command_promote)

    package = subparsers.add_parser("package")
    package.add_argument("--checkpoint", default="checkpoints/final_pipeline.pt")
    package.add_argument("--output", default="artifacts/pinna_submission.zip")
    package.set_defaults(function=command_package)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
