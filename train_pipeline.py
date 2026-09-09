"""Unified proposal-aligned pipeline CLI.

Run ``python train_pipeline.py <stage> --help`` for stage-specific arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
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
    BilateralEarLandmarkDataset,
    EarLandmarkDataset,
    EarLocatorDataset,
    EarMeshLandmarkDataset,
    prediction_key,
)
from src.pointnet2_model import default_model_config
from src.pointnext_model import default_pointnext_config
from src.pointtransformerv3_model import (
    PTV3_AMP_DTYPE,
    PTV3_SPCONV_ALGORITHM,
    PTV3_UPSTREAM_REVISION,
    default_pointtransformerv3_config,
    validate_pointtransformerv3_checkpoint_config,
)
from src.precision import autocast_context
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


def _json_default(value):
    """Convert NumPy values while keeping unsupported objects fail-fast."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def write_json(path: Path, data) -> None:
    """Atomically write JSON without leaving a truncated target on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                data,
                handle,
                default=_json_default,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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


def landmark_model_config(
    args,
    local_scale: float,
    curve_landmark_fractions=None,
) -> dict:
    decoder = str(getattr(args, "landmark_decoder", "coordinate-regression"))
    surface_decoder = decoder in {"surface-heatmap", "surface-curve"}
    bilateral_mode = str(getattr(args, "bilateral_mode", "none"))
    cascade_stages = int(getattr(args, "cascade_stages", 0))
    if bilateral_mode != "none" and (
        args.backbone != "pointnext" or decoder != "surface-heatmap"
    ):
        raise ValueError(
            "--bilateral-mode requires --backbone pointnext and "
            "--landmark-decoder surface-heatmap"
        )
    if (
        bilateral_mode == "landmark-cross-attention"
        and int(args.heatmap_feature_dim)
        % int(args.bilateral_attention_heads)
    ):
        raise ValueError(
            "--bilateral-attention-heads must divide --heatmap-feature-dim"
        )
    if cascade_stages:
        if args.backbone != "pointnext" or decoder != "surface-heatmap":
            raise ValueError(
                "--cascade-stages requires --backbone pointnext and "
                "--landmark-decoder surface-heatmap"
            )
        if bilateral_mode != "none":
            raise ValueError(
                "landmark-token cascade and --bilateral-mode are separate experiments"
            )
        if str(args.refinement_mode) != "geometry-offset":
            raise ValueError(
                "landmark-token cascade requires --refinement-mode geometry-offset"
            )
        if bool(getattr(args, "surface_voting", False)):
            raise ValueError(
                "landmark-token cascade and --surface-voting are separate experiments"
            )
        if int(args.heatmap_feature_dim) % int(args.cascade_attention_heads):
            raise ValueError(
                "--cascade-attention-heads must divide --heatmap-feature-dim"
            )
    if surface_decoder and args.backbone != "pointnext":
        raise ValueError(
            "--landmark-decoder surface-heatmap/surface-curve requires "
            "--backbone pointnext"
        )
    if not surface_decoder and (
        str(args.refinement_mode) != "geometry-offset"
        or int(args.refinement_stages) != 1
    ):
        raise ValueError(
            "feature-aware or multi-stage refinement requires "
            "--landmark-decoder surface-heatmap"
        )
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
    if args.backbone == "pointnet2":
        encoder = pointnet_encoder_config()
    elif args.backbone == "pointnext":
        encoder = default_pointnext_config(
            width=getattr(args, "pointnext_width", None),
            variant=str(getattr(args, "pointnext_variant", "s")),
        )
    elif args.backbone == "pointtransformerv3":
        if not getattr(args, "amp", True):
            raise ValueError(
                "Point Transformer V3 requires FP16 AMP; remove --no-amp"
            )
        encoder = default_pointtransformerv3_config(args.ptv3_grid_size)
    else:
        raise ValueError(f"unsupported landmark backbone: {args.backbone}")
    if surface_decoder:
        if int(args.heatmap_topk) > int(args.num_points):
            raise ValueError("--heatmap-topk cannot exceed --num-points")
        refinement_mode = str(args.refinement_mode)
        if refinement_mode == "feature-attention":
            if int(args.refinement_k) == 0:
                raise ValueError(
                    "--refinement-mode feature-attention requires --refinement-k"
                )
            if str(args.refinement_anchor) != "raw":
                raise ValueError(
                    "--refinement-mode feature-attention requires --refinement-anchor raw"
                )
        elif int(args.refinement_stages) != 1:
            raise ValueError(
                "--refinement-stages greater than one requires "
                "--refinement-mode feature-attention"
            )
        config = {
            "backbone": "pointnext",
            "decoder": decoder.replace("-", "_"),
            "encoder_config": encoder,
            "four_heads": bool(args.four_heads),
            "heatmap_feature_dim": int(args.heatmap_feature_dim),
            "heatmap_topk": int(args.heatmap_topk),
            "heatmap_coordinate_temperature": float(
                args.heatmap_coordinate_temperature
            ),
            "surface_voting": bool(getattr(args, "surface_voting", False)),
            "vote_cap_normalized": (
                float(getattr(args, "vote_cap_mm", 6.0)) / local_scale
                if bool(getattr(args, "surface_voting", False))
                else 0.0
            ),
            "vote_fusion_iterations": int(
                getattr(args, "vote_fusion_iterations", 3)
            ),
            "vote_fusion_epsilon_normalized": (
                float(getattr(args, "vote_fusion_epsilon_mm", 0.25))
                / local_scale
                if bool(getattr(args, "surface_voting", False))
                else 0.0
            ),
            "refinement_k": int(args.refinement_k),
            "refinement_cap_normalized": 5.0 / local_scale if args.refinement_k else 0.0,
            "refinement_anchor": str(getattr(args, "refinement_anchor", "raw")),
            "refinement_mode": refinement_mode,
            "refinement_stages": int(args.refinement_stages),
            "refinement_hidden_dim": int(args.refinement_hidden_dim),
            "refinement_temperature": float(args.refinement_temperature),
            "cascade_stages": cascade_stages,
            "cascade_attention_heads": int(args.cascade_attention_heads),
            "cascade_radius_normalized": (
                float(args.cascade_radius_mm) / local_scale
                if cascade_stages
                else 0.0
            ),
            "cascade_radius_decay": float(args.cascade_radius_decay),
            "cascade_dropout": float(args.cascade_dropout),
        }
        if decoder == "surface-curve":
            if curve_landmark_fractions is None:
                raise ValueError(
                    "surface-curve requires training-fold landmark fractions"
                )
            config.update(
                {
                    "curve_landmark_fractions": [
                        float(value) for value in curve_landmark_fractions
                    ],
                    "curve_logit_weight": float(args.curve_logit_weight),
                    "curve_arc_logit_weight": float(
                        args.curve_arc_logit_weight
                    ),
                    "curve_arc_temperature": float(
                        args.curve_arc_temperature
                    ),
                }
            )
        if bilateral_mode != "none":
            config.update(
                {
                    "bilateral_mode": bilateral_mode,
                    "bilateral_attention_heads": int(
                        args.bilateral_attention_heads
                    ),
                    "bilateral_attention_layers": int(
                        args.bilateral_attention_layers
                    ),
                    "bilateral_dropout": float(args.bilateral_dropout),
                }
            )
        return config
    config = {
        "backbone": args.backbone,
        "encoder_config": encoder,
        "four_heads": bool(args.four_heads),
        "head_channels": [512, 256],
        "dropout": 0.0,
        "refinement_k": int(args.refinement_k),
        "refinement_cap_normalized": 5.0 / local_scale if args.refinement_k else 0.0,
        "refinement_anchor": str(getattr(args, "refinement_anchor", "raw")),
    }
    if args.backbone == "pointtransformerv3":
        # Upstream spconv issue #563 identifies mixed-precision evaluation as
        # the failing path. The encoder fixes its sparse convolutions to the
        # Native algorithm, bypassing ConvTunerSimple; PTv3 retains FP16 AMP.
        config["amp_dtype"] = PTV3_AMP_DTYPE
    return config


def landmark_loss_config(args) -> dict:
    decoder = str(getattr(args, "landmark_decoder", "coordinate-regression"))
    surface_decoder = decoder in {"surface-heatmap", "surface-curve"}
    heatmap_weight = float(getattr(args, "heatmap_weight", 0.0))
    if surface_decoder and heatmap_weight <= 0.0:
        raise ValueError(
            "surface heatmap/curve training requires a positive --heatmap-weight"
        )
    if not surface_decoder and heatmap_weight != 0.0:
        raise ValueError(
            "--heatmap-weight is only valid with a surface heatmap/curve decoder"
        )
    heatmap_distance = str(getattr(args, "heatmap_distance", "euclidean"))
    geodesic_cache_dir = getattr(args, "geodesic_cache_dir", None)
    surface_voting = bool(getattr(args, "surface_voting", False))
    cascade_stages = int(getattr(args, "cascade_stages", 0))
    cascade_coordinate_weight = float(
        getattr(args, "cascade_coordinate_weight", 0.0)
    )
    cascade_heatmap_weight = float(
        getattr(args, "cascade_heatmap_weight", 0.0)
    )
    if cascade_stages:
        if cascade_coordinate_weight <= 0.0 or cascade_heatmap_weight <= 0.0:
            raise ValueError(
                "landmark-token cascade requires positive "
                "--cascade-coordinate-weight and --cascade-heatmap-weight"
            )
    elif cascade_coordinate_weight != 0.0 or cascade_heatmap_weight != 0.0:
        raise ValueError(
            "cascade auxiliary weights require --cascade-stages"
        )
    vote_weight = float(getattr(args, "vote_weight", 0.0))
    vote_radius_mm = float(getattr(args, "vote_radius_mm", 6.0))
    vote_cap_mm = float(getattr(args, "vote_cap_mm", 6.0))
    if decoder == "surface-curve" and heatmap_distance != "geodesic":
        raise ValueError(
            "--landmark-decoder surface-curve requires "
            "--heatmap-distance geodesic"
        )
    if not surface_decoder and heatmap_distance != "euclidean":
        raise ValueError(
            "--heatmap-distance geodesic requires a surface heatmap/curve decoder"
        )
    if heatmap_distance == "geodesic" and not geodesic_cache_dir:
        raise ValueError(
            "--heatmap-distance geodesic requires --geodesic-cache-dir"
        )
    if heatmap_distance != "geodesic" and geodesic_cache_dir:
        raise ValueError(
            "--geodesic-cache-dir is only valid with --heatmap-distance geodesic"
        )
    if surface_voting and (heatmap_distance != "geodesic" or vote_weight <= 0.0):
        raise ValueError(
            "--surface-voting requires geodesic heatmaps and positive --vote-weight"
        )
    if not surface_voting and vote_weight != 0.0:
        raise ValueError("--vote-weight requires --surface-voting")
    if surface_voting and vote_cap_mm < vote_radius_mm:
        raise ValueError("--vote-cap-mm must be at least --vote-radius-mm")
    curve_weight = float(getattr(args, "curve_weight", 0.0))
    curve_arc_weight = float(getattr(args, "curve_arc_weight", 0.0))
    if decoder == "surface-curve":
        if curve_weight <= 0.0 or curve_arc_weight <= 0.0:
            raise ValueError(
                "surface-curve training requires positive --curve-weight and "
                "--curve-arc-weight"
            )
        if surface_voting:
            raise ValueError(
                "surface-curve and --surface-voting are separate controlled experiments"
            )
    elif curve_weight != 0.0 or curve_arc_weight != 0.0:
        raise ValueError(
            "curve loss weights require --landmark-decoder surface-curve"
        )
    return {
        "anchor": float(args.anchor_weight),
        "spacing": float(args.spacing_weight),
        "surface": float(args.surface_weight),
        "heatmap": heatmap_weight,
        "heatmap_sigma_mm": float(getattr(args, "heatmap_sigma_mm", 2.0)),
        "heatmap_distance": heatmap_distance,
        "vote": vote_weight,
        "vote_radius_mm": vote_radius_mm,
        "vote_cap_mm": vote_cap_mm,
        "curve": curve_weight,
        "curve_arc": curve_arc_weight,
        "curve_sigma_mm": float(getattr(args, "curve_sigma_mm", 3.0)),
        "curve_arc_radius_mm": float(
            getattr(args, "curve_arc_radius_mm", 4.0)
        ),
        "cascade_coordinate": cascade_coordinate_weight,
        "cascade_heatmap": cascade_heatmap_weight,
        "cascade_heatmap_sigma_mm": float(
            getattr(args, "cascade_heatmap_sigma_mm", 2.0)
        ),
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
        geodesic_cache_dir=(
            getattr(args, "geodesic_cache_dir", None)
            if str(getattr(args, "heatmap_distance", "euclidean")) == "geodesic"
            else None
        ),
        include_curve_targets=(
            str(getattr(args, "landmark_decoder", "coordinate-regression"))
            == "surface-curve"
        ),
    )
    if args.backbone == "meshnet":
        gate = read_json(args.meshnet_gate_json)
        return EarMeshLandmarkDataset(
            **common, target_faces=int(gate["target_faces"])
        )
    if str(getattr(args, "bilateral_mode", "none")) != "none":
        return BilateralEarLandmarkDataset(**common)
    return EarLandmarkDataset(**common)


def training_curve_landmark_fractions(dataset: Dataset, subject_ids: Sequence[str]):
    """Fit the inference arc-fraction template using training ears only."""

    from src.curve import median_landmark_arc_fractions

    id_to_index = {
        dataset.get_identifier(index): index for index in range(len(dataset))
    }
    ears = []
    for subject_id in subject_ids:
        if subject_id not in id_to_index:
            raise ValueError(f"unknown curve-training subject: {subject_id}")
        _, left, right = dataset[id_to_index[subject_id]]
        ears.extend((left, right))
    return median_landmark_arc_fractions(ears)


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
    primary_complete_coverage = float(args.primary_complete_coverage)
    if not 0.0 < primary_complete_coverage <= 1.0:
        raise ValueError("--primary-complete-coverage must be in (0, 1]")
    primary_expansion = float(args.primary_expansion)
    if not 0.0 <= primary_expansion <= 1.0:
        raise ValueError("--primary-expansion must be in [0, 1]")
    preliminary = calibrate_directional_crops(
        records,
        primary_complete_coverage=primary_complete_coverage,
        primary_expansion=primary_expansion,
    )
    geometry = []
    for record in records:
        mesh, _, _ = dataset[index[record.subject_id]]
        canonical_prediction = canonicalize_xyz(record.predicted_center, record.ear)
        primary, _ = boxes_for_prediction(canonical_prediction, preliminary)
        crop = clip_mesh_to_box(mesh, primary.for_ear(record.ear))
        geometry.append(crop_geometry_stats(crop))
    calibration = calibrate_directional_crops(
        records,
        primary_complete_coverage=primary_complete_coverage,
        primary_expansion=primary_expansion,
        geometry_stats=geometry,
    )
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


def command_prepare_geodesic_targets(args):
    """Precompute fold-bound mesh-geodesic fields outside the training loop."""
    from src.geodesic import prepare_geodesic_cache

    folds = read_json(args.folds_json)
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    validate_fold_dataset(dataset, folds)
    if str(args.outer_fold) == "final":
        subject_ids = [
            dataset.get_identifier(index) for index in range(len(dataset))
        ]
    else:
        outer = select_outer_fold(folds, int(args.outer_fold))
        subject_ids = list(outer["train"]) + list(outer["validation"])
    manifest = prepare_geodesic_cache(
        dataset,
        subject_ids,
        _load_prediction_map(args.predictions_json),
        read_json(args.calibration_json),
        args.output_dir,
        args.folds_json,
        args.predictions_json,
        args.calibration_json,
        args.outer_fold,
    )
    print(
        f"Prepared {manifest['ear_count']} geodesic ear caches at "
        f"{args.output_dir}"
    )


_CASCADE_MODEL_CONFIG_KEYS = {
    "cascade_stages",
    "cascade_attention_heads",
    "cascade_radius_normalized",
    "cascade_radius_decay",
    "cascade_dropout",
}


def _initialize_cascade_from_checkpoint(
    model,
    checkpoint_path: str,
    model_config: Mapping[str, object],
    data_config: Mapping[str, object],
    loss_weights: Mapping[str, object],
) -> dict:
    """Load only a structurally and fold-identical non-cascade baseline."""

    if int(model_config.get("cascade_stages", 0)) <= 0:
        raise ValueError(
            "--initialize-from-checkpoint is only valid with --cascade-stages"
        )
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu")
    if (
        checkpoint.get("component_schema_version") != 1
        or checkpoint.get("component") != "landmarks"
    ):
        raise ValueError(
            "cascade initialization requires a component-schema-1 landmark checkpoint"
        )
    source_config = dict(checkpoint.get("model_config", {}))
    if int(source_config.get("cascade_stages", 0)) != 0:
        raise ValueError(
            "cascade initialization source must be the non-cascade baseline"
        )
    target_base = {
        key: value
        for key, value in model_config.items()
        if key not in _CASCADE_MODEL_CONFIG_KEYS
    }
    source_base = {
        key: value
        for key, value in source_config.items()
        if key not in _CASCADE_MODEL_CONFIG_KEYS
    }
    def same_serialized_value(left, right) -> bool:
        return json.dumps(
            left, sort_keys=True, default=_json_default
        ) == json.dumps(right, sort_keys=True, default=_json_default)

    config_mismatches = [
        key
        for key, value in source_base.items()
        if key not in target_base
        or not same_serialized_value(target_base[key], value)
    ]
    if config_mismatches:
        raise ValueError(
            "cascade initialization model does not match the requested baseline: "
            + ", ".join(sorted(config_mismatches))
        )

    source_data = dict(checkpoint.get("data_config", {}))
    for key in ("outer_fold", "train_ids", "validation_ids", "num_points"):
        if source_data.get(key) != data_config.get(key):
            raise ValueError(
                f"cascade initialization {key} does not match the current fold"
            )
    source_checksums = source_data.get("artifact_checksums")
    if not isinstance(source_checksums, Mapping):
        raise ValueError(
            "cascade initialization checkpoint lacks artifact checksums"
        )
    if dict(source_checksums) != dict(data_config["artifact_checksums"]):
        raise ValueError(
            "cascade initialization folds/predictions/calibration checksums differ"
        )
    source_losses = source_data.get("loss_weights")
    if not isinstance(source_losses, Mapping):
        raise ValueError(
            "cascade initialization checkpoint lacks its baseline loss configuration"
        )
    baseline_loss_keys = (
        "anchor",
        "spacing",
        "surface",
        "heatmap",
        "heatmap_sigma_mm",
        "heatmap_distance",
    )
    backward_loss_defaults = {"heatmap_distance": "euclidean"}
    loss_mismatches = [
        key
        for key in baseline_loss_keys
        if source_losses.get(key, backward_loss_defaults.get(key))
        != loss_weights.get(key)
    ]
    if loss_mismatches:
        raise ValueError(
            "cascade initialization baseline losses differ: "
            + ", ".join(loss_mismatches)
        )

    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("cascade_layers.")
    ]
    if unexpected or missing:
        raise ValueError(
            "cascade initialization state dictionary is incompatible; "
            f"unexpected={unexpected}, missing_non_cascade={missing}"
        )
    source_best = checkpoint.get("metrics", {}).get("best_md_mm")
    return {
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": file_sha256(path),
        "source_epoch": int(checkpoint.get("epoch", 0)),
        "source_best_md_mm": (
            float(source_best) if source_best is not None else None
        ),
        "loaded_non_cascade_state_key_count": len(
            checkpoint["model_state_dict"]
        ),
        "new_cascade_state_keys": list(incompatible.missing_keys),
    }


def command_fit_landmarks(args):
    seed_everything(args.seed)
    device = resolve_device(args.device)
    folds = read_json(args.folds_json)
    outer = select_outer_fold(folds, args.outer_fold)
    calibration = read_json(args.calibration_json)
    predictions = _load_prediction_map(args.predictions_json)
    source_dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    validate_fold_dataset(source_dataset, folds)
    geodesic_manifest = None
    if str(getattr(args, "heatmap_distance", "euclidean")) == "geodesic":
        from src.geodesic import validate_geodesic_manifest

        geodesic_manifest = validate_geodesic_manifest(
            args.geodesic_cache_dir,
            args.outer_fold,
            list(outer["train"]) + list(outer["validation"]),
            args.folds_json,
            args.predictions_json,
            args.calibration_json,
        )
    dense_points = 32768 if args.surface_weight else 0
    train_data = make_landmark_dataset(
        args, predictions, calibration, outer["train"], args.seed, True
    )
    validation_data = make_landmark_dataset(
        args, predictions, calibration, outer["validation"], args.seed + 100_000, False
    )
    curve_fractions = (
        training_curve_landmark_fractions(source_dataset, outer["train"])
        if str(getattr(args, "landmark_decoder", "coordinate-regression"))
        == "surface-curve"
        else None
    )
    model_config = landmark_model_config(
        args,
        float(calibration["local_scale"]),
        curve_landmark_fractions=curve_fractions,
    )
    model = make_landmark_model(model_config)
    loss_weights = landmark_loss_config(args)
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
        "bilateral_mode": str(getattr(args, "bilateral_mode", "none")),
        "loss_weights": loss_weights,
        "amp_dtype": model_config.get("amp_dtype", "auto"),
    }
    if geodesic_manifest is not None:
        data_config["geodesic_cache"] = {
            "path": str(args.geodesic_cache_dir),
            "manifest_sha256": file_sha256(
                Path(args.geodesic_cache_dir) / "manifest.json"
            ),
            "method": geodesic_manifest["method"],
        }
    if args.initialize_from_checkpoint:
        data_config["initialization"] = _initialize_cascade_from_checkpoint(
            model,
            args.initialize_from_checkpoint,
            model_config,
            data_config,
            loss_weights,
        )
    metrics = train_landmarks(
        model, train_data, validation_data, args.output_dir, model_config, data_config,
        loss_weights, device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        amp=args.amp,
        resume=not args.no_resume,
        amp_dtype=str(model_config.get("amp_dtype", "auto")),
        encoder_learning_rate=args.encoder_learning_rate,
        warmup_epochs=args.warmup_epochs,
        minimum_learning_rate=args.minimum_learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
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
    curve_path_enabled = bool(getattr(args, "curve_path_decode", False))
    if curve_path_enabled and str(
        landmark_checkpoint["model_config"].get("decoder", "")
    ) != "surface_curve":
        raise ValueError(
            "--curve-path-decode requires a final surface-curve landmark model"
        )
    curve_path_config = {
        "enabled": curve_path_enabled,
        "field_strength": float(
            getattr(args, "curve_path_field_strength", 4.0)
        ),
        "backtrack_weight": float(
            getattr(args, "curve_path_backtrack_weight", 8.0)
        ),
    }
    if curve_path_enabled:
        from src.curve import contour_anchor_manifest

        curve_path_config.update(
            {
                "routing": "section_anchors",
                "anchor_indices": contour_anchor_manifest(),
            }
        )
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
        "postprocess": {
            "project_to_surface": bool(args.project_to_surface),
            "curve_path_decoder": curve_path_config,
        },
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

    curve_fractions = (
        training_curve_landmark_fractions(dataset, subject_ids)
        if str(getattr(args, "landmark_decoder", "coordinate-regression"))
        == "surface-curve"
        else None
    )
    landmark_config = landmark_model_config(
        args,
        float(calibration["local_scale"]),
        curve_landmark_fractions=curve_fractions,
    )
    landmark_model = make_landmark_model(landmark_config)
    dense = 32768 if args.surface_weight else 0
    landmark_data = make_landmark_dataset(
        args, predictions, calibration, subject_ids, args.seed, True
    )
    weights = landmark_loss_config(args)
    train_landmarks(
        landmark_model, landmark_data, None, str(output / "landmarks"), landmark_config,
        {"calibration": calibration, "train_ids": subject_ids, "loss_weights": weights},
        weights, device,
        epochs=args.landmark_epochs,
        batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        amp=args.amp,
        resume=not args.no_resume,
        amp_dtype=str(landmark_config.get("amp_dtype", "auto")),
        encoder_learning_rate=args.encoder_learning_rate,
        warmup_epochs=args.warmup_epochs,
        minimum_learning_rate=args.minimum_learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
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
            "left": float(compute_mean_landmark_distance(left, left_target)),
            "right": float(compute_mean_landmark_distance(right, right_target)),
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


def command_evaluate_projection(args):
    """Compare raw and exact surface-projected outputs on one held-out fold."""
    # Reuse the strictly validated fold reconstruction used by the qualitative
    # viewer so evaluation and visualization cannot silently preprocess an ear
    # differently from landmark validation.
    from src.surface import project_points_to_mesh
    from visualize_point_importance import (
        EAR_NAMES,
        load_fold_context,
        prepare_trace,
    )

    context = load_fold_context(args)
    raw_errors = []
    projected_errors = []
    projection_displacements = []
    projection_timings_ms = []
    per_ear = {}
    total = len(context.validation_ids) * len(EAR_NAMES)
    completed = 0

    for subject_id in context.validation_ids:
        for ear in EAR_NAMES:
            trace = prepare_trace(
                context, subject_id, ear, include_projection=False
            )
            start = time.perf_counter()
            projected = project_points_to_mesh(
                trace.final_world, trace.prepared.crop_mesh
            )
            projection_timings_ms.append((time.perf_counter() - start) * 1000.0)
            projected = np.asarray(projected, dtype=np.float32)
            if projected.shape != (85, 3) or not np.isfinite(projected).all():
                raise RuntimeError(
                    f"surface projection failed for {subject_id}:{ear}"
                )

            raw = np.asarray(trace.raw_errors_mm, dtype=np.float64)
            projected_error = np.linalg.norm(
                projected.astype(np.float64)
                - trace.ground_truth_world.astype(np.float64),
                axis=1,
            )
            displacement = np.linalg.norm(
                projected.astype(np.float64)
                - trace.final_world.astype(np.float64),
                axis=1,
            )
            if (
                raw.shape != (85,)
                or projected_error.shape != (85,)
                or not np.isfinite(raw).all()
                or not np.isfinite(projected_error).all()
                or not np.isfinite(displacement).all()
            ):
                raise RuntimeError(
                    f"projection metrics are invalid for {subject_id}:{ear}"
                )

            raw_errors.append(raw)
            projected_errors.append(projected_error)
            projection_displacements.append(displacement)
            raw_md = float(np.mean(raw))
            projected_md = float(np.mean(projected_error))
            per_ear[prediction_key(subject_id, ear)] = {
                "raw_md_mm": raw_md,
                "projected_md_mm": projected_md,
                "delta_mm": projected_md - raw_md,
                "mean_projection_displacement_mm": float(np.mean(displacement)),
                "backup_triggered": bool(
                    trace.prepared.crop_stats.get("used_backup", False)
                ),
            }
            completed += 1
            print(
                f"Evaluated projection {completed}/{total}: "
                f"{subject_id}:{ear} raw={raw_md:.6f} mm "
                f"projected={projected_md:.6f} mm"
            )

    raw_array = np.stack(raw_errors)
    projected_array = np.stack(projected_errors)
    displacement_array = np.stack(projection_displacements)
    raw_ear_md = np.asarray(
        [item["raw_md_mm"] for item in per_ear.values()], dtype=np.float64
    )
    projected_ear_md = np.asarray(
        [item["projected_md_mm"] for item in per_ear.values()], dtype=np.float64
    )

    def distribution(values):
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90.0)),
            "p95": float(np.percentile(values, 95.0)),
            "maximum": float(np.max(values)),
        }

    raw_pooled = float(np.mean(raw_array))
    projected_pooled = float(np.mean(projected_array))
    delta = projected_pooled - raw_pooled
    report = {
        "schema_version": 1,
        "component": "fold_surface_projection_evaluation",
        "checkpoint_path": str(context.checkpoint_path),
        "checkpoint_sha256": file_sha256(context.checkpoint_path),
        "outer_fold": int(context.outer_fold),
        "run_seed": int(context.run_seed),
        "backbone": context.backbone,
        "provenance_level": context.provenance_level,
        "subject_count": len(context.validation_ids),
        "ear_count": len(per_ear),
        "raw": {
            "pooled_md_mm": raw_pooled,
            "per_landmark_md_mm": np.mean(raw_array, axis=0).tolist(),
            "ear_distribution_mm": distribution(raw_ear_md),
        },
        "projected": {
            "pooled_md_mm": projected_pooled,
            "per_landmark_md_mm": np.mean(projected_array, axis=0).tolist(),
            "ear_distribution_mm": distribution(projected_ear_md),
        },
        "comparison": {
            "delta_mm": delta,
            "relative_change_percent": (
                float(100.0 * delta / raw_pooled) if raw_pooled > 0 else 0.0
            ),
            "improved_ears": int(
                sum(
                    item["projected_md_mm"] < item["raw_md_mm"]
                    for item in per_ear.values()
                )
            ),
            "unchanged_ears": int(np.sum(projected_ear_md == raw_ear_md)),
            "worsened_ears": int(np.sum(projected_ear_md > raw_ear_md)),
            "mean_projection_displacement_mm": float(
                np.mean(displacement_array)
            ),
        },
        "runtime": {
            "projection_median_ms_per_ear": float(
                np.median(projection_timings_ms)
            ),
            "projection_mean_ms_per_ear": float(
                np.mean(projection_timings_ms)
            ),
        },
        "per_ear": per_ear,
    }
    write_json(Path(args.output), report)
    print(
        f"Fold {context.outer_fold} seed {context.run_seed}: "
        f"raw={raw_pooled:.6f} mm, projected={projected_pooled:.6f} mm, "
        f"delta={delta:+.6f} mm"
    )


def command_analyze_heatmap_decoder(args):
    """Re-decode one existing fold heatmap checkpoint without retraining."""
    from src.heatmap_diagnostics import main as analyze_heatmap

    values = [
        "--checkpoint-path",
        args.checkpoint_path,
        "--prior-path",
        args.prior_path,
        "--prior-manifest",
        args.prior_manifest,
        "--mesh-dir",
        args.mesh_dir,
        "--landmarks-dir",
        args.landmarks_dir,
        "--folds-json",
        args.folds_json,
        "--predictions-json",
        args.predictions_json,
        "--calibration-json",
        args.calibration_json,
        "--top-k",
        *(str(value) for value in args.top_k),
        "--temperatures",
        *(str(value) for value in args.temperatures),
        "--components",
        str(args.components),
        "--beta",
        str(args.beta),
        "--projection-workers",
        str(args.projection_workers),
        "--device",
        args.device,
        "--output",
        args.output,
    ]
    if args.run_seed is not None:
        values.extend(["--run-seed", str(args.run_seed)])
    analyze_heatmap(values)


def command_analyze_geodesic_candidates(args):
    """Measure whether the existing heatmap retrieves correct surface candidates."""
    from src.geodesic_diagnostics import main as analyze_geodesic

    values = [
        "--checkpoint-path", args.checkpoint_path,
        "--mesh-dir", args.mesh_dir,
        "--landmarks-dir", args.landmarks_dir,
        "--folds-json", args.folds_json,
        "--predictions-json", args.predictions_json,
        "--calibration-json", args.calibration_json,
        "--geodesic-cache-dir", args.geodesic_cache_dir,
        "--top-k", *(str(value) for value in args.top_k),
        "--euclidean-close-mm", str(args.euclidean_close_mm),
        "--geodesic-far-mm", str(args.geodesic_far_mm),
        "--normal-dot-threshold", str(args.normal_dot_threshold),
        "--device", args.device,
        "--output", args.output,
    ]
    if args.run_seed is not None:
        values.extend(["--run-seed", str(args.run_seed)])
    analyze_geodesic(values)


def command_generate_pca_prior(args):
    """Fit a leakage-safe PCA prior using only one outer-training fold."""
    from src.shape_prior.generate_prior import main as generate_prior

    generate_prior(
        [
            "--mesh-dir",
            args.mesh_dir,
            "--landmarks-dir",
            args.landmarks_dir,
            "--folds-json",
            args.folds_json,
            "--outer-fold",
            str(args.outer_fold),
            "--predictions-json",
            args.predictions_json,
            "--calibration-json",
            args.calibration_json,
            "--components",
            str(args.components),
            "--beta",
            str(args.beta),
            "--output",
            args.output,
            "--manifest",
            args.manifest,
        ]
    )


def command_evaluate_pca_prior(args):
    """Evaluate PCA before exact projection on a strictly held-out fold."""
    from src.shape_prior.evaluate_prior import main as evaluate_prior

    values = [
        "--checkpoint-path",
        args.checkpoint_path,
        "--prior-path",
        args.prior_path,
        "--prior-manifest",
        args.prior_manifest,
        "--mesh-dir",
        args.mesh_dir,
        "--landmarks-dir",
        args.landmarks_dir,
        "--folds-json",
        args.folds_json,
        "--predictions-json",
        args.predictions_json,
        "--calibration-json",
        args.calibration_json,
        "--components",
        str(args.components),
        "--beta",
        str(args.beta),
        "--output",
        args.output,
        "--device",
        args.device,
    ]
    if args.run_seed is not None:
        values.extend(["--run-seed", str(args.run_seed)])
    if args.curve_path_decode:
        values.extend(
            [
                "--curve-path-decode",
                "--curve-path-field-strength",
                str(args.curve_path_field_strength),
                "--curve-path-backtrack-weight",
                str(args.curve_path_backtrack_weight),
            ]
        )
    evaluate_prior(values)


def command_summarize_pca_prior(args):
    from src.shape_prior.summarize_prior import main as summarize_prior

    summarize_prior(
        [
            "--report-root",
            args.report_root,
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--output",
            args.output,
        ]
    )


def command_generate_bilateral_pca_prior(args):
    """Fit paired common-morphology and signed-asymmetry PCA bases."""
    from src.shape_prior.generate_bilateral_prior import main as generate_prior

    generate_prior(
        [
            "--mesh-dir",
            args.mesh_dir,
            "--landmarks-dir",
            args.landmarks_dir,
            "--folds-json",
            args.folds_json,
            "--outer-fold",
            str(args.outer_fold),
            "--predictions-json",
            args.predictions_json,
            "--calibration-json",
            args.calibration_json,
            "--common-components",
            str(args.common_components),
            "--asymmetry-components",
            str(args.asymmetry_components),
            "--common-beta",
            str(args.common_beta),
            "--asymmetry-beta",
            str(args.asymmetry_beta),
            "--output",
            args.output,
            "--manifest",
            args.manifest,
        ]
    )


def command_evaluate_bilateral_pca_prior(args):
    """Screen paired PCA settings or run one exact projected confirmation."""
    from src.shape_prior.evaluate_bilateral_prior import main as evaluate_prior

    values = [
        "--checkpoint-path",
        args.checkpoint_path,
        "--prior-path",
        args.prior_path,
        "--prior-manifest",
        args.prior_manifest,
        "--reference-report",
        args.reference_report,
        "--mesh-dir",
        args.mesh_dir,
        "--landmarks-dir",
        args.landmarks_dir,
        "--folds-json",
        args.folds_json,
        "--predictions-json",
        args.predictions_json,
        "--calibration-json",
        args.calibration_json,
        "--common-components",
        *(str(value) for value in args.common_components),
        "--asymmetry-components",
        *(str(value) for value in args.asymmetry_components),
        "--common-betas",
        *(str(value) for value in args.common_betas),
        "--asymmetry-betas",
        *(str(value) for value in args.asymmetry_betas),
        "--output",
        args.output,
        "--device",
        args.device,
    ]
    if args.independent_prior_path is not None:
        values.extend(
            ["--independent-prior-path", args.independent_prior_path]
        )
    if args.contour_gate:
        values.extend(["--contour-gate", *args.contour_gate])
    if args.skip_projection:
        values.append("--skip-projection")
    if args.run_seed is not None:
        values.extend(["--run-seed", str(args.run_seed)])
    evaluate_prior(values)


def command_summarize_bilateral_pca_prior(args):
    from src.shape_prior.summarize_bilateral_prior import main as summarize_prior

    summarize_prior(
        [
            "--report-root",
            args.report_root,
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--output",
            args.output,
        ]
    )


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_ptv3_preflight(args):
    """Fail-fast H200 gate for the exact optional PTv3 dependency stack."""

    output = Path(args.output)
    device = resolve_device(args.device)
    report = {
        "schema_version": 1,
        "component": "pointtransformerv3_preflight",
        "passed": False,
        "upstream_revision": PTV3_UPSTREAM_REVISION,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": bool(
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            "distributions": {
                name: _installed_version(name)
                for name in (
                    "addict",
                    "timm",
                    "spconv-cu118",
                    "torch-scatter",
                    "flash-attn",
                )
            },
        },
        "required_configuration": {
            "num_points": int(args.num_points),
            "grid_sizes": [float(value) for value in args.grid_size],
            "flash_attention": True,
            "autocast_dtype": PTV3_AMP_DTYPE,
            "gradient_scaling_in_training": True,
            "spconv_algorithm": PTV3_SPCONV_ALGORITHM,
            "spconv_issue_workaround": "ConvAlgo.Native",
            "full_encoder_decoder": True,
            "global_pool": "max",
            "refinement_k": 32,
        },
        "expected_environment": {
            "python": "3.10",
            "torch": "2.1.0",
            "torch_cuda": "11.8",
            "distributions": {
                "addict": "2.4.0",
                "timm": "0.9.16",
                "spconv-cu118": "2.3.8",
                "torch-scatter": "2.1.2",
                "flash-attn": "2.5.9.post1",
            },
        },
        "grid_results": [],
    }
    try:
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError(
                f"PTv3 requires Python 3.10; found {sys.version_info.major}.{sys.version_info.minor}"
            )
        if torch.__version__.split("+")[0] != "2.1.0":
            raise RuntimeError(f"PTv3 requires torch 2.1.0; found {torch.__version__}")
        if torch.version.cuda != "11.8":
            raise RuntimeError(
                f"PTv3 requires the CUDA 11.8 Torch build; found {torch.version.cuda}"
            )
        expected_distributions = report["expected_environment"]["distributions"]
        mismatches = {
            name: report["environment"]["distributions"][name]
            for name, expected in expected_distributions.items()
            if report["environment"]["distributions"][name] is None
            or not report["environment"]["distributions"][name].startswith(expected)
        }
        if mismatches:
            raise RuntimeError(
                "PTv3 dependency versions are missing or incompatible: "
                f"{mismatches}; expected {expected_distributions}"
            )
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("PTv3 preflight requires an allocated CUDA GPU")
        report["environment"]["device"] = str(device)
        report["environment"]["gpu"] = torch.cuda.get_device_name(device)
        for module_name in (
            "addict",
            "timm",
            "spconv.pytorch",
            "torch_scatter",
            "flash_attn",
        ):
            importlib.import_module(module_name)

        seed_everything(args.seed)
        xyz = torch.rand(
            1, int(args.num_points), 3, device=device, dtype=torch.float32
        ) * 1.8 - 0.9
        normals = torch.randn(
            1, int(args.num_points), 3, device=device, dtype=torch.float32
        )
        normals = torch.nn.functional.normalize(normals, dim=-1)
        fixed_input = torch.cat([xyz, normals], dim=-1)

        for grid_size in args.grid_size:
            encoder_config = default_pointtransformerv3_config(float(grid_size))
            model_config = {
                "backbone": "pointtransformerv3",
                "encoder_config": encoder_config,
                "four_heads": True,
                "head_channels": [512, 256],
                "dropout": 0.0,
                "refinement_k": 32,
                "refinement_cap_normalized": 0.125,
                "amp_dtype": PTV3_AMP_DTYPE,
            }
            model = make_landmark_model(model_config).to(device)
            probe_optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            probe_scaler = torch.cuda.amp.GradScaler(enabled=True)
            train_input = fixed_input.detach().clone().requires_grad_(True)
            torch.cuda.reset_peak_memory_stats(device)
            model.train()
            started = time.perf_counter()
            with autocast_context(device, True, PTV3_AMP_DTYPE):
                training_output = model(train_input)
                loss = training_output.float().square().mean()
            probe_scaler.scale(loss).backward()
            probe_scaler.unscale_(probe_optimizer)
            torch.cuda.synchronize(device)
            train_ms = (time.perf_counter() - started) * 1000.0
            training_counts = list(model.encoder.last_voxel_counts)
            finite_training = bool(torch.isfinite(training_output).all().item())
            finite_input_gradient = bool(
                train_input.grad is not None
                and torch.isfinite(train_input.grad).all().item()
            )
            finite_parameter_gradients = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )

            model.eval()
            with torch.no_grad(), autocast_context(device, True, PTV3_AMP_DTYPE):
                first = model(fixed_input)
                first_counts = list(model.encoder.last_voxel_counts)
                second = model(fixed_input)
                second_counts = list(model.encoder.last_voxel_counts)
            deterministic = bool(torch.equal(first, second))
            finite_evaluation = bool(torch.isfinite(first).all().item())

            # Deployment starts in a fresh process and enters eval directly,
            # without a preceding training pass to populate spconv caches.
            fresh_model = make_landmark_model(model_config).to(device)
            fresh_model.load_state_dict(model.state_dict())
            fresh_model.eval()
            with torch.no_grad(), autocast_context(device, True, PTV3_AMP_DTYPE):
                fresh_first = fresh_model(fixed_input)
                fresh_first_counts = list(fresh_model.encoder.last_voxel_counts)
                fresh_second = fresh_model(fixed_input)
                fresh_second_counts = list(fresh_model.encoder.last_voxel_counts)
            deterministic_fresh_evaluation = bool(
                torch.equal(fresh_first, fresh_second)
            )
            reload_matches_evaluation = bool(torch.equal(first, fresh_first))
            finite_fresh_evaluation = bool(torch.isfinite(fresh_first).all().item())
            result = {
                "grid_size": float(grid_size),
                "input_points": int(args.num_points),
                "training_voxel_counts": training_counts,
                "evaluation_voxel_counts": first_counts,
                "repeat_voxel_counts": second_counts,
                "fresh_evaluation_voxel_counts": fresh_first_counts,
                "fresh_repeat_voxel_counts": fresh_second_counts,
                "output_shape": list(first.shape),
                "finite_training_output": finite_training,
                "finite_input_gradient": finite_input_gradient,
                "finite_parameter_gradients": finite_parameter_gradients,
                "grad_scaler_enabled": bool(probe_scaler.is_enabled()),
                "spconv_algorithm": model.encoder.spconv_algorithm,
                "spconv_layer_count": int(model.encoder.spconv_layer_count),
                "expected_spconv_layer_count": int(
                    model.encoder.expected_spconv_layer_count
                ),
                "finite_evaluation_output": finite_evaluation,
                "deterministic_evaluation": deterministic,
                "finite_fresh_evaluation_output": finite_fresh_evaluation,
                "deterministic_fresh_evaluation": deterministic_fresh_evaluation,
                "fresh_reload_matches_evaluation": reload_matches_evaluation,
                "forward_backward_ms": train_ms,
                "peak_gpu_memory_mb": float(
                    torch.cuda.max_memory_allocated(device) / 1024**2
                ),
            }
            result["passed"] = bool(
                finite_training
                and finite_input_gradient
                and finite_parameter_gradients
                and probe_scaler.is_enabled()
                and model.encoder.spconv_algorithm == PTV3_SPCONV_ALGORITHM
                and model.encoder.spconv_layer_count
                == model.encoder.expected_spconv_layer_count
                and finite_evaluation
                and deterministic
                and finite_fresh_evaluation
                and deterministic_fresh_evaluation
                and reload_matches_evaluation
                and tuple(first.shape) == (1, 85, 3)
                and training_counts
                == first_counts
                == second_counts
                == fresh_first_counts
                == fresh_second_counts
            )
            report["grid_results"].append(result)
            del (
                model,
                fresh_model,
                probe_optimizer,
                probe_scaler,
                train_input,
                training_output,
                first,
                second,
                fresh_first,
                fresh_second,
                loss,
            )
            torch.cuda.empty_cache()

        report["passed"] = bool(
            report["grid_results"]
            and all(item["passed"] for item in report["grid_results"])
        )
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def command_package(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 2:
        raise ValueError("package requires a complete schema-version-2 pipeline checkpoint")
    required = {"locator", "landmark", "broad_config", "crop_calibration", "coordinates", "sampling", "postprocess"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"v2 checkpoint is incomplete: {missing}")
    landmark_config = checkpoint.get("landmark", {}).get("model_config", {})
    include_ptv3 = landmark_config.get("backbone") == "pointtransformerv3"
    if include_ptv3:
        validate_pointtransformerv3_checkpoint_config(landmark_config)
    root = Path(__file__).resolve().parent
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    package_paths = [
        root / "src",
        root / "configs",
        root / "__init__.py",
        root / "requirements.txt",
        root / "README.md",
        root / "TECHNICAL_METHOD.md",
        root / "RESEARCH_BASIS.md",
        root / "THIRD_PARTY_NOTICES.md",
    ]
    if include_ptv3:
        package_paths.extend(
            [root / "requirements-ptv3.txt", root / "PTV3_EXPERIMENT.md"]
        )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_paths:
            if path.is_dir():
                for child in path.rglob("*"):
                    relative = child.relative_to(root)
                    is_ptv3_source = (
                        relative == Path("src/pointtransformerv3_model.py")
                        or Path("src/third_party/pointtransformerv3") in relative.parents
                    )
                    if is_ptv3_source and not include_ptv3:
                        continue
                    if not child.is_file() or (
                        child.suffix not in {".py", ".json"}
                        and child.name != "LICENSE"
                    ):
                        continue
                    archive.write(child, relative)
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


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative finite number"
        )
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1]")
    return parsed


def signed_unit_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [-1, 1]")
    return parsed


def add_landmark_training_arguments(parser):
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1e-3,
        help="AdamW rate for landmark heads/refiner and, by default, the encoder",
    )
    parser.add_argument(
        "--encoder-learning-rate",
        type=positive_float,
        default=None,
        help="optional separate AdamW rate for the point encoder",
    )
    parser.add_argument(
        "--weight-decay",
        type=nonnegative_float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=nonnegative_integer,
        default=0,
        help="linear warm-up from 10%% to the configured rates before cosine decay",
    )
    parser.add_argument(
        "--minimum-learning-rate",
        type=nonnegative_float,
        default=0.0,
        help="absolute final rate for every AdamW parameter group",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=nonnegative_float,
        default=0.0,
        help="maximum global gradient norm; zero disables clipping",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=positive_integer,
        default=32,
        help="target batch size reached through gradient accumulation",
    )


def add_landmark_model_arguments(parser):
    parser.add_argument(
        "--backbone",
        choices=("pointnet2", "pointnext", "pointtransformerv3", "meshnet"),
        default="pointnet2",
    )
    parser.add_argument(
        "--ptv3-grid-size",
        type=float,
        choices=(0.01, 0.02),
        default=0.01,
        help="normalized PTv3 voxel size; ignored by other backbones",
    )
    parser.add_argument(
        "--pointnext-width",
        type=positive_integer,
        default=None,
        help=(
            "optional PointNeXt base-width override; by default S/B/L use "
            "C32 and XL uses C64; ignored by other backbones"
        ),
    )
    parser.add_argument(
        "--pointnext-variant",
        choices=("s", "b", "l", "xl"),
        default="s",
        help=(
            "PointNeXt depth preset: s=[1,1,1,1,1], b=[1,2,3,2,2], "
            "l=[1,3,5,3,3], xl=[1,4,7,4,4]; ignored by other backbones"
        ),
    )
    parser.add_argument(
        "--landmark-decoder",
        choices=("coordinate-regression", "surface-heatmap", "surface-curve"),
        default="coordinate-regression",
        help=(
            "coordinate-regression preserves the legacy global XYZ heads; "
            "surface-heatmap retains PointNeXt spatial features and predicts "
            "85 distributions over sampled crop-surface points; surface-curve "
            "adds shared contour-membership and ordered arc-coordinate fields"
        ),
    )
    parser.add_argument(
        "--heatmap-feature-dim",
        type=positive_integer,
        default=128,
        help="full-resolution point/query feature width for the surface decoder",
    )
    parser.add_argument(
        "--heatmap-topk",
        type=positive_integer,
        default=64,
        help="highest-scoring surface candidates used for each coordinate expectation",
    )
    parser.add_argument(
        "--heatmap-coordinate-temperature",
        type=positive_float,
        default=1.0,
        help="softmax temperature within the selected surface candidates",
    )
    parser.add_argument(
        "--heatmap-weight",
        type=nonnegative_float,
        default=0.0,
        help="weight of Gaussian surface-heatmap KL; must be positive for the surface decoder",
    )
    parser.add_argument(
        "--heatmap-sigma-mm",
        type=positive_float,
        default=2.0,
        help="Gaussian target standard deviation in original millimetres",
    )
    parser.add_argument(
        "--cascade-stages",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "full-surface landmark-token residual heatmap stages; zero "
            "preserves the established decoder exactly"
        ),
    )
    parser.add_argument(
        "--cascade-attention-heads",
        type=positive_integer,
        default=8,
        help="within-ear landmark-token self-attention heads",
    )
    parser.add_argument(
        "--cascade-radius-mm",
        type=positive_float,
        default=8.0,
        help="first-stage soft spatial-attention radius in original millimetres",
    )
    parser.add_argument(
        "--cascade-radius-decay",
        type=unit_float,
        default=0.5,
        help="multiplicative soft-radius reduction for each later cascade stage",
    )
    parser.add_argument(
        "--cascade-dropout",
        type=unit_float,
        default=0.0,
        help="landmark-token attention/MLP dropout",
    )
    parser.add_argument(
        "--cascade-coordinate-weight",
        type=nonnegative_float,
        default=0.0,
        help="intermediate coordinate-MD supervision weight",
    )
    parser.add_argument(
        "--cascade-heatmap-weight",
        type=nonnegative_float,
        default=0.0,
        help="intermediate surface-heatmap supervision weight",
    )
    parser.add_argument(
        "--cascade-heatmap-sigma-mm",
        type=positive_float,
        default=2.0,
        help="Gaussian width for pre-cascade heatmap supervision",
    )
    parser.add_argument(
        "--heatmap-distance",
        choices=("euclidean", "geodesic"),
        default="euclidean",
        help=(
            "distance used to construct heatmap targets; geodesic requires a "
            "fold-bound cache prepared with prepare-geodesic-targets"
        ),
    )
    parser.add_argument(
        "--geodesic-cache-dir",
        help="directory containing the exact fold geodesic manifest and ear caches",
    )
    parser.add_argument(
        "--surface-voting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="decode heatmap candidates after bounded learned surface-vector votes",
    )
    parser.add_argument(
        "--vote-weight",
        type=nonnegative_float,
        default=0.0,
        help="weight of local candidate-to-landmark vector supervision",
    )
    parser.add_argument(
        "--vote-radius-mm",
        type=positive_float,
        default=6.0,
        help="maximum mesh-geodesic distance of candidates supervised as voters",
    )
    parser.add_argument(
        "--vote-cap-mm",
        type=positive_float,
        default=6.0,
        help="tanh bound on the norm of each predicted candidate vote offset",
    )
    parser.add_argument(
        "--vote-fusion-iterations",
        type=positive_integer,
        default=3,
        help="deterministic weighted geometric-median iterations over Top-K votes",
    )
    parser.add_argument(
        "--vote-fusion-epsilon-mm",
        type=positive_float,
        default=0.25,
        help="minimum residual in robust vote reweighting",
    )
    parser.add_argument(
        "--curve-weight",
        type=nonnegative_float,
        default=0.0,
        help="weight of four dense geodesic contour-field KL losses",
    )
    parser.add_argument(
        "--curve-arc-weight",
        type=nonnegative_float,
        default=0.0,
        help="weight of normalized within-contour arc-coordinate supervision",
    )
    parser.add_argument(
        "--curve-sigma-mm",
        type=positive_float,
        default=3.0,
        help="Gaussian width of dense anatomical-contour targets",
    )
    parser.add_argument(
        "--curve-arc-radius-mm",
        type=positive_float,
        default=4.0,
        help="geodesic tube radius supervised by the arc-coordinate loss",
    )
    parser.add_argument(
        "--curve-logit-weight",
        type=nonnegative_float,
        default=0.5,
        help="contour-field contribution to structured landmark logits",
    )
    parser.add_argument(
        "--curve-arc-logit-weight",
        type=nonnegative_float,
        default=0.25,
        help="ordered arc-coordinate contribution to structured landmark logits",
    )
    parser.add_argument(
        "--curve-arc-temperature",
        type=positive_float,
        default=0.1,
        help="normalized arc-coordinate Gaussian width during landmark decoding",
    )
    parser.add_argument(
        "--bilateral-mode",
        choices=("none", "shared-latent", "landmark-cross-attention"),
        default="none",
        help=(
            "optional paired-ear PointNeXt heatmap experiment; 'none' keeps "
            "the established independent-ear pipeline unchanged"
        ),
    )
    parser.add_argument(
        "--bilateral-attention-heads",
        type=positive_integer,
        default=8,
        help="cross-attention heads; used only by landmark-cross-attention",
    )
    parser.add_argument(
        "--bilateral-attention-layers",
        type=int,
        choices=(1, 2),
        default=1,
        help="paired landmark-token blocks; used only by landmark-cross-attention",
    )
    parser.add_argument(
        "--bilateral-dropout",
        type=unit_float,
        default=0.0,
        help="bilateral attention/MLP dropout; zero preserves deterministic evaluation",
    )
    parser.add_argument("--meshnet-gate-json")
    parser.add_argument("--four-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refinement-k", type=int, choices=(0, 32, 64), default=0)
    parser.add_argument(
        "--refinement-anchor",
        choices=("raw", "nearest-surface-sample"),
        default="raw",
        help=(
            "KNN query centre for local refinement; nearest-surface-sample "
            "anchors the query to the closest sampled input point"
        ),
    )
    parser.add_argument(
        "--refinement-mode",
        choices=("geometry-offset", "feature-attention"),
        default="geometry-offset",
        help=(
            "geometry-offset preserves the legacy KNN max-pool refiner; "
            "feature-attention uses decoded PointNeXt features, heatmap "
            "confidence, normals, and iterative local surface attention"
        ),
    )
    parser.add_argument(
        "--refinement-stages",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of feature-attention surface refinement iterations",
    )
    parser.add_argument(
        "--refinement-hidden-dim",
        type=positive_integer,
        default=128,
        help="hidden width of the feature-attention surface refiner",
    )
    parser.add_argument(
        "--refinement-temperature",
        type=positive_float,
        default=1.0,
        help="local candidate softmax temperature for feature-attention refinement",
    )
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
    calibrate.add_argument(
        "--primary-complete-coverage",
        type=float,
        default=0.99,
        help=(
            "training-OOF complete-ear coverage target used to select the smallest "
            "0-5 mm primary-crop safety margin (default: 0.99)"
        ),
    )
    calibrate.add_argument(
        "--primary-expansion",
        type=float,
        default=0.0,
        help=(
            "post-safety directional expansion applied to the primary crop; "
            "0.2 promotes the proposal's first backup reach to primary "
            "(default: 0.0)"
        ),
    )
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

    geodesic_targets = subparsers.add_parser("prepare-geodesic-targets")
    add_data_arguments(geodesic_targets)
    geodesic_targets.add_argument("--folds-json", required=True)
    geodesic_targets.add_argument(
        "--outer-fold", required=True, help="0-4 for a fold cache or final"
    )
    geodesic_targets.add_argument("--predictions-json", required=True)
    geodesic_targets.add_argument("--calibration-json", required=True)
    geodesic_targets.add_argument("--output-dir", required=True)
    geodesic_targets.set_defaults(function=command_prepare_geodesic_targets)

    landmarks = subparsers.add_parser("fit-landmarks")
    add_data_arguments(landmarks)
    add_runtime_arguments(landmarks)
    add_landmark_training_arguments(landmarks)
    add_landmark_model_arguments(landmarks)
    landmarks.add_argument("--folds-json", required=True)
    landmarks.add_argument("--outer-fold", type=int, required=True)
    landmarks.add_argument("--predictions-json", required=True)
    landmarks.add_argument("--calibration-json", required=True)
    landmarks.add_argument("--output-dir", required=True)
    landmarks.add_argument(
        "--initialize-from-checkpoint",
        help=(
            "optional matching non-cascade best_landmarks.pt used to initialize "
            "the established backbone/decoder/refiner weights"
        ),
    )
    landmarks.set_defaults(function=command_fit_landmarks)

    final = subparsers.add_parser("fit-final")
    add_data_arguments(final)
    add_runtime_arguments(final)
    add_landmark_training_arguments(final)
    add_landmark_model_arguments(final)
    final.add_argument("--locator-run-root", required=True)
    final.add_argument("--calibration-json", required=True)
    final.add_argument("--locator-epochs", type=int, required=True)
    final.add_argument("--landmark-epochs", type=int, required=True)
    final.add_argument("--project-to-surface", action=argparse.BooleanOptionalAction, default=False)
    final.add_argument(
        "--curve-path-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="embed deterministic connected-curve mesh-path decoding",
    )
    final.add_argument(
        "--curve-path-field-strength", type=nonnegative_float, default=4.0
    )
    final.add_argument(
        "--curve-path-backtrack-weight", type=nonnegative_float, default=8.0
    )
    final.add_argument("--output-dir", default="checkpoints")
    final.set_defaults(function=command_fit_final)

    evaluate = subparsers.add_parser("evaluate")
    add_data_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(function=command_evaluate)

    projection = subparsers.add_parser("evaluate-projection")
    add_data_arguments(projection)
    projection.add_argument("--checkpoint-path", required=True)
    projection.add_argument("--predictions-json", required=True)
    projection.add_argument("--calibration-json", required=True)
    projection.add_argument("--folds-json", required=True)
    projection.add_argument("--run-seed", type=int)
    projection.add_argument("--output", required=True)
    projection.add_argument("--device", default="auto")
    projection.set_defaults(function=command_evaluate_projection)

    heatmap_diagnostic = subparsers.add_parser("analyze-heatmap-decoder")
    add_data_arguments(heatmap_diagnostic)
    heatmap_diagnostic.add_argument("--checkpoint-path", required=True)
    heatmap_diagnostic.add_argument("--prior-path", required=True)
    heatmap_diagnostic.add_argument("--prior-manifest", required=True)
    heatmap_diagnostic.add_argument("--folds-json", required=True)
    heatmap_diagnostic.add_argument("--predictions-json", required=True)
    heatmap_diagnostic.add_argument("--calibration-json", required=True)
    heatmap_diagnostic.add_argument(
        "--top-k", nargs="+", type=positive_integer, required=True
    )
    heatmap_diagnostic.add_argument(
        "--temperatures", nargs="+", type=positive_float, required=True
    )
    heatmap_diagnostic.add_argument("--components", type=positive_integer, default=32)
    heatmap_diagnostic.add_argument("--beta", type=unit_float, default=0.5)
    heatmap_diagnostic.add_argument("--run-seed", type=int)
    heatmap_diagnostic.add_argument(
        "--projection-workers", type=positive_integer, default=10
    )
    heatmap_diagnostic.add_argument("--device", default="auto")
    heatmap_diagnostic.add_argument("--output", required=True)
    heatmap_diagnostic.set_defaults(function=command_analyze_heatmap_decoder)

    geodesic_diagnostic = subparsers.add_parser("analyze-geodesic-candidates")
    add_data_arguments(geodesic_diagnostic)
    geodesic_diagnostic.add_argument("--checkpoint-path", required=True)
    geodesic_diagnostic.add_argument("--folds-json", required=True)
    geodesic_diagnostic.add_argument("--predictions-json", required=True)
    geodesic_diagnostic.add_argument("--calibration-json", required=True)
    geodesic_diagnostic.add_argument("--geodesic-cache-dir", required=True)
    geodesic_diagnostic.add_argument(
        "--top-k", nargs="+", type=positive_integer, default=[1, 8, 32, 64]
    )
    geodesic_diagnostic.add_argument(
        "--euclidean-close-mm", type=positive_float, default=4.0
    )
    geodesic_diagnostic.add_argument(
        "--geodesic-far-mm", type=positive_float, default=8.0
    )
    geodesic_diagnostic.add_argument(
        "--normal-dot-threshold", type=signed_unit_float, default=0.0
    )
    geodesic_diagnostic.add_argument("--run-seed", type=int)
    geodesic_diagnostic.add_argument("--device", default="auto")
    geodesic_diagnostic.add_argument("--output", required=True)
    geodesic_diagnostic.set_defaults(function=command_analyze_geodesic_candidates)

    pca_generate = subparsers.add_parser("generate-pca-prior")
    add_data_arguments(pca_generate)
    pca_generate.add_argument("--folds-json", required=True)
    pca_generate.add_argument("--outer-fold", required=True, help="0-4 or final")
    pca_generate.add_argument("--predictions-json", required=True)
    pca_generate.add_argument("--calibration-json", required=True)
    pca_generate.add_argument("--components", type=positive_integer, default=32)
    pca_generate.add_argument("--beta", type=unit_float, default=1.0)
    pca_generate.add_argument("--output", required=True)
    pca_generate.add_argument("--manifest", required=True)
    pca_generate.set_defaults(function=command_generate_pca_prior)

    pca_evaluate = subparsers.add_parser("evaluate-pca-prior")
    add_data_arguments(pca_evaluate)
    pca_evaluate.add_argument("--checkpoint-path", required=True)
    pca_evaluate.add_argument("--prior-path", required=True)
    pca_evaluate.add_argument("--prior-manifest", required=True)
    pca_evaluate.add_argument("--folds-json", required=True)
    pca_evaluate.add_argument("--predictions-json", required=True)
    pca_evaluate.add_argument("--calibration-json", required=True)
    pca_evaluate.add_argument("--components", type=positive_integer, default=32)
    pca_evaluate.add_argument("--beta", type=unit_float, default=1.0)
    pca_evaluate.add_argument("--run-seed", type=int)
    pca_evaluate.add_argument("--output", required=True)
    pca_evaluate.add_argument("--device", default="auto")
    pca_evaluate.add_argument(
        "--curve-path-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    pca_evaluate.add_argument(
        "--curve-path-field-strength", type=nonnegative_float, default=4.0
    )
    pca_evaluate.add_argument(
        "--curve-path-backtrack-weight", type=nonnegative_float, default=8.0
    )
    pca_evaluate.set_defaults(function=command_evaluate_pca_prior)

    pca_summary = subparsers.add_parser("summarize-pca-prior")
    pca_summary.add_argument("--report-root", required=True)
    pca_summary.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    pca_summary.add_argument("--output", required=True)
    pca_summary.set_defaults(function=command_summarize_pca_prior)

    bilateral_pca_generate = subparsers.add_parser(
        "generate-bilateral-pca-prior"
    )
    add_data_arguments(bilateral_pca_generate)
    bilateral_pca_generate.add_argument("--folds-json", required=True)
    bilateral_pca_generate.add_argument(
        "--outer-fold", required=True, help="0-4 or final"
    )
    bilateral_pca_generate.add_argument("--predictions-json", required=True)
    bilateral_pca_generate.add_argument("--calibration-json", required=True)
    bilateral_pca_generate.add_argument(
        "--common-components", type=positive_integer, default=64
    )
    bilateral_pca_generate.add_argument(
        "--asymmetry-components", type=positive_integer, default=32
    )
    bilateral_pca_generate.add_argument(
        "--common-beta", type=unit_float, default=1.0
    )
    bilateral_pca_generate.add_argument(
        "--asymmetry-beta", type=unit_float, default=1.0
    )
    bilateral_pca_generate.add_argument("--output", required=True)
    bilateral_pca_generate.add_argument("--manifest", required=True)
    bilateral_pca_generate.set_defaults(
        function=command_generate_bilateral_pca_prior
    )

    bilateral_pca_evaluate = subparsers.add_parser(
        "evaluate-bilateral-pca-prior"
    )
    add_data_arguments(bilateral_pca_evaluate)
    bilateral_pca_evaluate.add_argument("--checkpoint-path", required=True)
    bilateral_pca_evaluate.add_argument("--prior-path", required=True)
    bilateral_pca_evaluate.add_argument("--prior-manifest", required=True)
    bilateral_pca_evaluate.add_argument("--reference-report", required=True)
    bilateral_pca_evaluate.add_argument("--independent-prior-path")
    bilateral_pca_evaluate.add_argument("--folds-json", required=True)
    bilateral_pca_evaluate.add_argument("--predictions-json", required=True)
    bilateral_pca_evaluate.add_argument("--calibration-json", required=True)
    bilateral_pca_evaluate.add_argument(
        "--common-components",
        nargs="+",
        type=positive_integer,
        required=True,
    )
    bilateral_pca_evaluate.add_argument(
        "--asymmetry-components",
        nargs="+",
        type=positive_integer,
        required=True,
    )
    bilateral_pca_evaluate.add_argument(
        "--common-betas", nargs="+", type=unit_float, required=True
    )
    bilateral_pca_evaluate.add_argument(
        "--asymmetry-betas", nargs="+", type=unit_float, required=True
    )
    bilateral_pca_evaluate.add_argument(
        "--contour-gate",
        nargs="+",
        choices=(
            "outer_helix",
            "concha",
            "inner_helix",
            "superior_antihelix",
        ),
    )
    bilateral_pca_evaluate.add_argument(
        "--skip-projection", action="store_true"
    )
    bilateral_pca_evaluate.add_argument("--run-seed", type=int)
    bilateral_pca_evaluate.add_argument("--output", required=True)
    bilateral_pca_evaluate.add_argument("--device", default="auto")
    bilateral_pca_evaluate.set_defaults(
        function=command_evaluate_bilateral_pca_prior
    )

    bilateral_pca_summary = subparsers.add_parser(
        "summarize-bilateral-pca-prior"
    )
    bilateral_pca_summary.add_argument("--report-root", required=True)
    bilateral_pca_summary.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 43, 44]
    )
    bilateral_pca_summary.add_argument("--output", required=True)
    bilateral_pca_summary.set_defaults(
        function=command_summarize_bilateral_pca_prior
    )

    gate = subparsers.add_parser("meshnet-gate")
    add_data_arguments(gate)
    gate.add_argument("--predictions-json", required=True)
    gate.add_argument("--calibration-json", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--seed", type=int, default=42)
    gate.set_defaults(function=command_meshnet_gate)

    ptv3 = subparsers.add_parser("ptv3-preflight")
    ptv3.add_argument("--output", default="artifacts/ptv3/preflight.json")
    ptv3.add_argument("--num-points", type=int, default=16384)
    ptv3.add_argument(
        "--grid-size",
        type=float,
        nargs="+",
        choices=(0.01, 0.02),
        default=[0.01, 0.02],
    )
    ptv3.add_argument("--seed", type=int, default=42)
    ptv3.add_argument("--device", default="auto")
    ptv3.set_defaults(function=command_ptv3_preflight)

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
