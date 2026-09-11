"""Export held-out heatmap predictions and anatomical-frame error components.

Ground truth is used only to measure errors and define diagnostic frames. It
never changes model inputs, candidate selection, PCA, or surface projection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from .dataset import Dataset
from .heatmap_diagnostics import _metric_summary, _positive_float, _positive_int, _unit_float
from .pipeline_dataset import EAR_NAMES, prediction_key, prepare_ear_geometry
from .precision import checkpoint_autocast_context
from .proposal_models import PointNeXtSurfaceHeatmapRegressor
from .shape_prior.evaluate_prior import (
    CONTOURS, _device, _load_model, _validate_context, _validate_prior_manifest,
    _world_from_local,
)
from .shape_prior.fitting import file_sha256, load_center_predictions, read_json
from .shape_prior.pca import PCAShapePrior
from .surface import project_points_to_mesh_with_faces
from .surface_geometry_features import surface_geometry_from_model_config


STAGES = ("coarse", "raw", "pca", "projected", "pca_projected")
COMPONENTS = ("along_contour", "across_contour", "surface_normal")
REGIONS = {name: (start, end) for name, start, end in CONTOURS}
REGIONS.update({"inner_helix_55_64": (55, 65), "inner_helix_65_74": (65, 75)})


def _unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(length, 1e-12), length[..., 0]


def anatomical_frames(target_world, normal_world, surface_distance_mm, ear,
                      max_surface_distance_mm=0.5):
    """Build orthonormal frames in the mirrored-left canonical frame.

    The tangent is the bisector of forward-oriented adjacent unit chords,
    projected into the target triangle's tangent plane. Contour endpoints use
    one-sided chords. Never join different contours or invent a fallback axis.
    Invalid rows are NaN and must be excluded only from directional summaries.
    """
    target = np.asarray(target_world, dtype=np.float64)
    normal = np.asarray(normal_world, dtype=np.float64)
    distance = np.asarray(surface_distance_mm, dtype=np.float64)
    if target.shape != (85, 3) or normal.shape != (85, 3) or distance.shape != (85,):
        raise ValueError("frames require (85, 3) targets/normals and (85,) distances")
    if not all(np.isfinite(v).all() for v in (target, normal, distance)):
        raise ValueError("frame inputs must be finite")
    if ear not in EAR_NAMES or not np.isfinite(max_surface_distance_mm) or max_surface_distance_mm <= 0:
        raise ValueError("invalid ear or frame surface-distance threshold")
    if np.any(distance < 0):
        raise ValueError("surface distances must be nonnegative")
    reflection = np.array([1.0, -1.0 if ear == "right" else 1.0, 1.0])
    target = target * reflection
    normal, normal_length = _unit_rows(normal * reflection)
    chord = np.zeros_like(target)
    valid_chord = np.zeros(85, dtype=bool)
    for _, start, end in CONTOURS:
        steps, lengths = _unit_rows(np.diff(target[start:end], axis=0))
        chord[start], chord[end - 1] = steps[0], steps[-1]
        chord[start + 1:end - 1] = steps[:-1] + steps[1:]
        valid_chord[start], valid_chord[end - 1] = lengths[0] > 1e-8, lengths[-1] > 1e-8
        valid_chord[start + 1:end - 1] = (lengths[:-1] > 1e-8) & (lengths[1:] > 1e-8)
    chord, chord_length = _unit_rows(chord)
    in_plane = chord - np.sum(chord * normal, axis=1, keepdims=True) * normal
    tangent, in_plane_length = _unit_rows(in_plane)
    across, _ = _unit_rows(np.cross(normal, tangent))
    valid_geometry = (normal_length > 1e-8) & valid_chord & (chord_length > 1e-8) & (in_plane_length > 1e-6)
    valid_surface = distance <= max_surface_distance_mm
    valid = valid_geometry & valid_surface
    basis = np.stack([tangent, across, normal], axis=1)
    if valid.any():
        gram = basis[valid] @ np.swapaxes(basis[valid], -1, -2)
        if not np.allclose(gram, np.eye(3), atol=1e-8, rtol=0):
            raise RuntimeError("diagnostic frames are not orthonormal")
    basis[~valid] = np.nan
    return basis, valid, {
        "degenerate_geometry": ~valid_geometry,
        "target_far_from_crop_surface": ~valid_surface,
        "tangent_plane_projection_length": in_plane_length,
    }


def decompose_errors(prediction_world, target_world, basis_canonical, valid, ear):
    prediction = np.asarray(prediction_world, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    if prediction.shape != (85, 3) or target.shape != (85, 3):
        raise ValueError("predictions and targets must have shape (85, 3)")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("predictions and targets must be finite")
    if ear not in EAR_NAMES:
        raise ValueError("invalid ear")
    error = prediction - target
    reflection = np.array([1.0, -1.0 if ear == "right" else 1.0, 1.0])
    signed = np.einsum("lij,lj->li", basis_canonical, error * reflection)
    squared = np.sum(error * error, axis=1)
    residual = np.abs(np.sum(signed[valid] ** 2, axis=1) - squared[valid])
    if not np.allclose(np.sum(signed[valid] ** 2, axis=1), squared[valid], atol=1e-8, rtol=1e-8):
        raise RuntimeError("directional components do not conserve squared error")
    return np.sqrt(squared), signed, float(residual.max()) if residual.size else 0.0


def component_summary(signed, valid):
    values = np.asarray(signed, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("valid error components must be finite triples")
    if not len(values):
        return {"valid_landmarks": 0, "components": None}
    energy = np.sum(values * values, axis=0)
    total = float(energy.sum())
    return {
        "valid_landmarks": len(values),
        "md_on_valid_frames_mm": float(np.linalg.norm(values, axis=1).mean()),
        "components": {
            name: {
                "mean_signed_mm": float(values[:, index].mean()),
                "mean_absolute_mm": float(np.abs(values[:, index]).mean()),
                "rms_mm": float(np.sqrt(np.mean(values[:, index] ** 2))),
                "squared_error_fraction": float(energy[index] / total) if total > 0 else None,
            }
            for index, name in enumerate(COMPONENTS)
        },
    }


def _stage_summary(errors, signed, valid):
    return {
        **_metric_summary(errors),
        "decomposition": component_summary(signed, valid),
        "decomposition_per_region": {
            name: component_summary(signed[:, start:end], valid[:, start:end])
            for name, (start, end) in REGIONS.items()
        },
        "decomposition_per_landmark": [
            component_summary(signed[:, index], valid[:, index]) for index in range(85)
        ],
    }


def _project_with_workers(points, mesh, executor, workers):
    chunks = np.array_split(points, min(workers, len(points)))
    results = list(executor.map(lambda chunk: project_points_to_mesh_with_faces(chunk, mesh), chunks))
    return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])


def _validate_reference(reference, args, fold, seed, validation_ids):
    expected = {
        "checkpoint_sha256": file_sha256(args.checkpoint_path),
        "prior_sha256": file_sha256(args.prior_path),
        "outer_fold": fold, "run_seed": seed,
        "components": args.components, "beta": args.beta,
        "ear_count": len(validation_ids) * 2,
    }
    for key, value in expected.items():
        if reference.get(key) != value:
            raise ValueError(f"reference PCA report {key} does not match the requested diagnostic")
    subjects = reference.get("subject_level_raw_and_pca_errors", {})
    if set(subjects) != set(validation_ids) or any(set(pair) != set(EAR_NAMES) for pair in subjects.values()):
        raise ValueError("reference PCA report does not contain exactly the held-out ear pairs")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-path", "prior-path", "prior-manifest", "mesh-dir", "landmarks-dir",
                 "folds-json", "predictions-json", "calibration-json", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--components", type=_positive_int, default=32)
    parser.add_argument("--beta", type=_unit_float, default=0.5)
    parser.add_argument("--run-seed", type=int)
    parser.add_argument("--projection-workers", type=_positive_int, default=10)
    parser.add_argument("--frame-max-surface-distance-mm", type=_positive_float, default=0.5)
    parser.add_argument("--reference-report")
    parser.add_argument("--reference-tolerance-mm", type=_positive_float, default=1e-4)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None):
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.suffix.lower() != ".json":
        raise ValueError("--output must end in .json; coordinate export uses the same stem with .npz")
    export = output.with_suffix(".npz")
    input_paths = [args.checkpoint_path, args.prior_path, args.prior_manifest, args.folds_json,
                   args.predictions_json, args.calibration_json, args.reference_report]
    if any(Path(p).resolve() in {output.resolve(), export.resolve()} for p in input_paths if p):
        raise ValueError("diagnostic output must not overwrite an input artifact")
    device = _device(args.device)
    checkpoint, model, model_config, data_config = _load_model(args.checkpoint_path, device)
    del checkpoint
    if not isinstance(model, PointNeXtSurfaceHeatmapRegressor) or model_config.get("decoder") != "surface_heatmap":
        raise ValueError("analyze-landmark-errors requires a single-ear PointNeXt surface-heatmap checkpoint")
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    dataset_ids = [dataset.get_identifier(i) for i in range(len(dataset))]
    subject_index = {subject: i for i, subject in enumerate(dataset_ids)}
    validation_ids, seed, num_points = _validate_context(args, data_config, dataset)
    fold = int(data_config["outer_fold"])
    calibration = read_json(args.calibration_json)
    centers = load_center_predictions(args.predictions_json, dataset_ids)
    prior = PCAShapePrior.load(args.prior_path)
    _validate_prior_manifest(read_json(args.prior_manifest), args, fold, data_config["train_ids"], prior)
    reference = read_json(args.reference_report) if args.reference_report else None
    if reference is not None:
        _validate_reference(reference, args, fold, seed, validation_ids)
    world_rows = {stage: [] for stage in STAGES}
    error_rows = {stage: [] for stage in STAGES}
    signed_rows = {stage: [] for stage in STAGES}
    targets, bases, masks, distances, surfaces, face_rows = [], [], [], [], [], []
    degenerate_rows, distant_rows, tangent_projection_rows = [], [], []
    subject_rows, ear_rows, sample_seeds = [], [], []
    per_ear, input_hashes = {}, {}
    max_energy_residual = 0.0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.projection_workers) as executor:
        for position, subject in enumerate(validation_ids):
            mesh, left, right = dataset[subject_index[subject]]
            input_hashes[subject] = {"mesh_sha256": file_sha256(Path(args.mesh_dir) / f"{subject}.ply")}
            for offset, ear in enumerate(EAR_NAMES):
                target = left if ear == "left" else right
                input_hashes[subject][f"{ear}_annotation_sha256"] = file_sha256(
                    Path(args.landmarks_dir) / f"{subject}_{ear}_landmarks.csv"
                )
                sample_seed = seed + 100_000 + (position * 2 + offset) * 1009
                prepared = prepare_ear_geometry(
                    mesh, target, ear, centers[prediction_key(subject, ear)], calibration,
                    num_points, sample_seed,
                    surface_geometry_config=surface_geometry_from_model_config(model_config),
                )
                tensor = torch.from_numpy(prepared.point_features).unsqueeze(0).to(device)
                with torch.no_grad(), checkpoint_autocast_context(device, model_config):
                    details = model.forward_with_details(tensor)
                local = {"coarse": details["coarse"][0].float().cpu().numpy(),
                         "raw": details["final"][0].float().cpu().numpy()}
                local["pca"] = prior.blend(local["raw"], beta=args.beta, n_components=args.components)
                world = {name: _world_from_local(prepared, ear, values) for name, values in local.items()}
                projected, faces = _project_with_workers(
                    np.concatenate([target, world["raw"], world["pca"]]),
                    prepared.crop_mesh, executor, args.projection_workers,
                )
                target_surface, world["projected"], world["pca_projected"] = np.split(projected, 3)
                target_faces = faces[:85]
                triangles = np.asarray(prepared.crop_mesh.vertices, dtype=np.float64)[
                    np.asarray(prepared.crop_mesh.faces)[target_faces]
                ]
                normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                surface_distance = np.linalg.norm(target_surface.astype(np.float64) - target, axis=1)
                basis, valid, quality = anatomical_frames(
                    target, normals, surface_distance, ear, args.frame_max_surface_distance_mm,
                )
                stages_for_ear = {}
                for stage in STAGES:
                    errors, signed, residual = decompose_errors(world[stage], target, basis, valid, ear)
                    world_rows[stage].append(world[stage])
                    error_rows[stage].append(errors)
                    signed_rows[stage].append(signed)
                    max_energy_residual = max(max_energy_residual, residual)
                    stages_for_ear[stage] = {"md_mm": float(errors.mean()),
                                           "decomposition": component_summary(signed, valid)}
                per_ear[prediction_key(subject, ear)] = {
                    "stages": stages_for_ear, "valid_frames": int(valid.sum()),
                    "degenerate_geometry_count": int(quality["degenerate_geometry"].sum()),
                    "target_far_from_crop_surface_count": int(quality["target_far_from_crop_surface"].sum()),
                    "target_to_crop_surface_mean_mm": float(surface_distance.mean()),
                    "target_to_crop_surface_max_mm": float(surface_distance.max()),
                    "sample_seed": sample_seed, "crop_stats": prepared.crop_stats,
                }
                targets.append(target); bases.append(basis); masks.append(valid)
                surfaces.append(target_surface); distances.append(surface_distance); face_rows.append(target_faces)
                degenerate_rows.append(quality["degenerate_geometry"])
                distant_rows.append(quality["target_far_from_crop_surface"])
                tangent_projection_rows.append(quality["tangent_plane_projection_length"])
                subject_rows.append(subject); ear_rows.append(ear); sample_seeds.append(sample_seed)
                print(f"Analyzed landmark errors {len(targets)}/{len(validation_ids)*2}: {subject}:{ear}", flush=True)
                del details, tensor
    valid = np.stack(masks)
    summaries = {stage: _stage_summary(error_rows[stage], np.stack(signed_rows[stage]), valid) for stage in STAGES}
    reproduction = None
    if reference is not None:
        deltas = []
        for subject, ear in zip(subject_rows, ear_rows):
            actual = per_ear[prediction_key(subject, ear)]["stages"]["pca_projected"]["md_mm"]
            expected = reference["subject_level_raw_and_pca_errors"][subject][ear]["pca_projected_md_mm"]
            deltas.append(abs(actual - expected))
        mean_delta = abs(summaries["pca_projected"]["pooled_md_mm"] - reference["pca_projected_mean_md_mm"])
        reproduction = {"passed": max(max(deltas), mean_delta) <= args.reference_tolerance_mm,
                        "max_absolute_per_ear_delta_mm": max(deltas),
                        "absolute_pooled_delta_mm": mean_delta, "tolerance_mm": args.reference_tolerance_mm}
    report = {
        "schema_version": 1, "component": "fold_landmark_error_decomposition",
        "outer_fold": fold, "run_seed": seed, "subject_count": len(validation_ids), "ear_count": len(targets),
        "validation_ids": validation_ids, "model_config": model_config, "num_points": num_points,
        "components": args.components, "beta": args.beta, "device": str(device),
        "projection_workers": args.projection_workers, "validation_sample_seed_offset": 100_000,
        "postprocess": "independent_pca_then_exact_crop_surface_projection",
        "frame_definition": {
            "coordinates": "canonical_left_ear_xyz_mm; right Y reflected before constructing basis",
            "normal": "exact closest crop triangle face normal at ground-truth landmark",
            "tangent": "adjacent forward unit-chord bisector projected into normal tangent plane; one-sided at contour endpoints",
            "across": "cross(normal, tangent) in canonical frame",
            "component_order": list(COMPONENTS), "error_sign": "prediction minus ground_truth",
            "max_target_to_surface_distance_mm": args.frame_max_surface_distance_mm,
            "invalid_frames": "excluded only from decomposition; exported as NaN and explicit mask; MD includes every landmark",
            "interpretation": "local directional description, not geodesic distance or proof of wrong-sheet prediction",
        },
        "frame_quality": {"valid_landmarks": int(valid.sum()), "total_landmarks": int(valid.size),
                          "valid_count_per_landmark": valid.sum(axis=0).tolist(),
                          "max_squared_error_identity_residual_mm2": max_energy_residual},
        "stages": summaries, "per_ear": per_ear, "input_subject_hashes": input_hashes,
        "reference_reproduction": reproduction,
        "caveat": "Squared component fractions partition squared error, not official mean Euclidean distance. Ground-truth frames are diagnostic only.",
        "runtime_seconds": float(time.perf_counter() - started),
    }
    for key in ("checkpoint_path", "prior_path", "prior_manifest", "folds_json", "predictions_json", "calibration_json", "reference_report"):
        path = getattr(args, key)
        if path:
            report[key] = str(path)
            hash_key = {"checkpoint_path": "checkpoint_sha256", "prior_path": "prior_sha256"}.get(key, f"{key}_sha256")
            report[hash_key] = file_sha256(path)
    arrays = {
        "schema_version": np.asarray(1), "subject_ids": np.asarray(subject_rows), "ears": np.asarray(ear_rows),
        "landmark_indices": np.arange(85), "sample_seeds": np.asarray(sample_seeds),
        "ground_truth_world_mm": np.stack(targets), "target_surface_world_mm": np.stack(surfaces),
        "target_crop_face_indices": np.stack(face_rows), "target_surface_distance_mm": np.stack(distances),
        "basis_canonical": np.stack(bases), "frame_valid": valid,
        "frame_degenerate_geometry": np.stack(degenerate_rows), "frame_target_far_from_surface": np.stack(distant_rows),
        "tangent_plane_projection_length": np.stack(tangent_projection_rows),
        "metadata_json": np.asarray(json.dumps({k: v for k, v in report.items() if k not in {"stages", "per_ear"}}, allow_nan=False)),
    }
    for stage in STAGES:
        arrays[f"{stage}_world_mm"] = np.stack(world_rows[stage])
        arrays[f"{stage}_error_mm"] = np.stack(error_rows[stage])
        arrays[f"{stage}_signed_components_mm"] = np.stack(signed_rows[stage])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(export, **arrays)
    report["coordinate_export"] = {"path": str(export), "sha256": file_sha256(export),
                                   "arrays": {key: list(value.shape) for key, value in arrays.items()},
                                   "loading": "numpy.load(path, allow_pickle=False)"}
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PCA-projected MD: {summaries['pca_projected']['pooled_md_mm']:.6f} mm")
    print(f"Valid anatomical frames: {valid.sum()}/{valid.size}")
    print(f"Report: {output}\nCoordinates: {export}")
    if reproduction is not None:
        print(f"Reference reproduction: {reproduction}")
        if not reproduction["passed"]:
            raise SystemExit("Reference reproduction failed; inspect the saved JSON before interpreting decomposition.")


if __name__ == "__main__":
    main()
