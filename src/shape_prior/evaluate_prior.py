"""Evaluate a saved PCA prior against one proposal fold checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from ..canonical import decanonicalize_xyz
from ..dataset import Dataset
from ..meshnet import meshnet_inputs_with_mesh
from ..pipeline_dataset import EAR_NAMES, prediction_key, prepare_ear_geometry
from ..pointtransformerv3_model import validate_pointtransformerv3_checkpoint_config
from ..precision import checkpoint_autocast_context
from ..proposal_models import build_fold_landmark_model
from .fitting import file_sha256, load_center_predictions, read_json
from .pca import PCAShapePrior


CONTOURS = (
    ("outer_helix", 0, 25),
    ("concha", 25, 55),
    ("inner_helix", 55, 75),
    ("superior_antihelix", 75, 85),
)


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def _torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _clean_state_dict(state_dict):
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _find_outer_fold(folds: Mapping[str, object], outer_fold: int) -> Mapping[str, object]:
    matches = [item for item in folds.get("outer", []) if int(item.get("fold", -1)) == outer_fold]
    if len(matches) != 1:
        raise ValueError(f"folds JSON does not contain exactly one outer fold {outer_fold}")
    return matches[0]


def _load_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = _torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("expected a proposal landmark checkpoint")
    if checkpoint.get("component_schema_version") != 1 or checkpoint.get("component") != "landmarks":
        raise ValueError("expected component_schema_version=1 and component='landmarks'")
    for key in ("model_state_dict", "model_config", "data_config"):
        if key not in checkpoint:
            raise ValueError(f"landmark checkpoint is missing {key}")
    model_config = dict(checkpoint["model_config"])
    if model_config.get("backbone") == "pointtransformerv3":
        validate_pointtransformerv3_checkpoint_config(model_config)
    model = build_fold_landmark_model(model_config).to(device)
    model.load_state_dict(_clean_state_dict(checkpoint["model_state_dict"]))
    model.eval()
    return checkpoint, model, model_config, dict(checkpoint["data_config"])


def _validate_context(args, data_config: Mapping[str, object], dataset: Dataset) -> tuple[list[str], int, int]:
    folds = read_json(args.folds_json)
    subject_ids = [dataset.get_identifier(index) for index in range(len(dataset))]
    checksum = "\n".join(sorted(subject_ids)).encode("utf-8")
    if folds.get("subject_checksum") != hashlib.sha256(checksum).hexdigest():
        raise ValueError("dataset subject IDs do not match folds JSON checksum")
    outer_fold = int(data_config.get("outer_fold", -1))
    fold = _find_outer_fold(folds, outer_fold)
    if list(data_config.get("train_ids", [])) != list(fold.get("train", [])):
        raise ValueError("checkpoint training IDs do not exactly match folds JSON")
    if list(data_config.get("validation_ids", [])) != list(fold.get("validation", [])):
        raise ValueError("checkpoint validation IDs do not exactly match folds JSON")
    calibration = read_json(args.calibration_json)
    if data_config.get("calibration") != calibration:
        raise ValueError("external calibration JSON does not match checkpoint calibration")
    hashes = data_config.get("artifact_checksums")
    if hashes is not None:
        expected = {
            "folds_json_sha256": file_sha256(args.folds_json),
            "predictions_json_sha256": file_sha256(args.predictions_json),
            "calibration_json_sha256": file_sha256(args.calibration_json),
        }
        if dict(hashes) != expected:
            raise ValueError("one or more checkpoint artifact hashes do not match")
    saved_seed = data_config.get("seed")
    if saved_seed is None and args.run_seed is None:
        raise ValueError("checkpoint does not store seed; provide --run-seed")
    if saved_seed is not None and args.run_seed is not None and int(saved_seed) != int(args.run_seed):
        raise ValueError(f"--run-seed {args.run_seed} does not match checkpoint seed {saved_seed}")
    return list(fold["validation"]), int(saved_seed if saved_seed is not None else args.run_seed), int(data_config["num_points"])


def _predict_local(model, model_config, backbone: str, prepared, ear: str, device: torch.device):
    if backbone == "meshnet":
        features, neighbors, _ = meshnet_inputs_with_mesh(
            prepared.crop_mesh,
            int(model_config["target_faces"]),
            ear,
            prepared.transform,
        )
        values = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(device)
        neighbor_values = torch.from_numpy(neighbors).unsqueeze(0).to(device)
        with torch.no_grad(), checkpoint_autocast_context(device, model_config):
            return model.forward_with_details(values, neighbor_values)["final"].squeeze(0).float().cpu().numpy()
    values = torch.from_numpy(prepared.point_features.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad(), checkpoint_autocast_context(device, model_config):
        return model.forward_with_details(values)["final"].squeeze(0).float().cpu().numpy()


def _world_from_local(prepared, ear: str, local: np.ndarray) -> np.ndarray:
    return decanonicalize_xyz(prepared.transform.denormalize_xyz(local), ear).astype(np.float32)


def _part_means(errors: np.ndarray) -> dict:
    return {name: float(errors[:, start:end].mean()) for name, start, end in CONTOURS}


def _write_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--prior-path", required=True)
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-seed", type=int)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = _device(args.device)
    checkpoint, model, model_config, data_config = _load_model(args.checkpoint_path, device)
    del checkpoint
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    subject_index = {dataset.get_identifier(index): index for index in range(len(dataset))}
    validation_ids, run_seed, num_points = _validate_context(args, data_config, dataset)
    calibration = read_json(args.calibration_json)
    predictions = load_center_predictions(args.predictions_json)
    prior = PCAShapePrior.load(args.prior_path)
    backbone = str(model_config.get("backbone", ""))

    ear_rows = []
    raw_errors = []
    pca_errors = []
    subjects = {}
    for subject_id in validation_ids:
        mesh, left, right = dataset[subject_index[subject_id]]
        subjects[subject_id] = {}
        for ear in EAR_NAMES:
            item = validation_ids.index(subject_id) * 2 + EAR_NAMES.index(ear)
            ground_truth = left if ear == "left" else right
            center = predictions[prediction_key(subject_id, ear)]
            prepared = prepare_ear_geometry(mesh, ground_truth, ear, center, calibration, num_points, run_seed + 100_000 + item * 1009)
            raw_local = _predict_local(model, model_config, backbone, prepared, ear, device)
            pca_local = prior.blend(raw_local)
            raw_world = _world_from_local(prepared, ear, raw_local)
            pca_world = _world_from_local(prepared, ear, pca_local)
            raw_landmark_errors = np.linalg.norm(raw_world - ground_truth, axis=1).astype(np.float32)
            pca_landmark_errors = np.linalg.norm(pca_world - ground_truth, axis=1).astype(np.float32)
            raw_md = float(raw_landmark_errors.mean())
            pca_md = float(pca_landmark_errors.mean())
            raw_errors.append(raw_landmark_errors)
            pca_errors.append(pca_landmark_errors)
            row = {
                "raw_md_mm": raw_md,
                "pca_md_mm": pca_md,
                "improvement_mm": raw_md - pca_md,
            }
            ear_rows.append(row)
            subjects[subject_id][ear] = row

    raw_ear = np.asarray([row["raw_md_mm"] for row in ear_rows], dtype=np.float32)
    pca_ear = np.asarray([row["pca_md_mm"] for row in ear_rows], dtype=np.float32)
    raw_stack = np.stack(raw_errors)
    pca_stack = np.stack(pca_errors)
    report = {
        "raw_mean_md_mm": float(raw_ear.mean()),
        "pca_mean_md_mm": float(pca_ear.mean()),
        "mean_improvement_mm": float(raw_ear.mean() - pca_ear.mean()),
        "raw_worst_ear_md_mm": float(raw_ear.max()),
        "pca_worst_ear_md_mm": float(pca_ear.max()),
        "worst_ear_improvement_mm": float(raw_ear.max() - pca_ear.max()),
        "ears_improved": int(np.sum(pca_ear < raw_ear)),
        "ears_unchanged": int(np.sum(pca_ear == raw_ear)),
        "ears_worsened": int(np.sum(pca_ear > raw_ear)),
        "raw_per_part_md_mm": _part_means(raw_stack),
        "pca_per_part_md_mm": _part_means(pca_stack),
        "subject_level_raw_and_pca_errors": subjects,
    }
    _write_json(args.output, report)
    print(
        f"Raw {report['raw_mean_md_mm']:.6f} mm, PCA {report['pca_mean_md_mm']:.6f} mm, "
        f"improvement {report['mean_improvement_mm']:.6f} mm; "
        f"improved {report['ears_improved']}/{len(ear_rows)}, worsened {report['ears_worsened']}"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
