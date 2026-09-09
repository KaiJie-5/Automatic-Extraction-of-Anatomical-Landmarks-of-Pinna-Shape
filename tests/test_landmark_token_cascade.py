import pytest
import torch

from src.losses import proposal_landmark_loss
from src.pointnext_model import default_pointnext_config
from src.proposal_models import (
    PointNeXtSurfaceHeatmapRegressor,
    build_landmark_model,
)
from train_pipeline import (
    _initialize_cascade_from_checkpoint,
    build_parser,
    landmark_loss_config,
    landmark_model_config,
)


def _tiny_encoder_config():
    return {
        **default_pointnext_config(width=8, variant="s"),
        "strides": [1, 2, 2, 2, 2],
        "nsample": 8,
    }


def test_one_stage_cascade_preserves_initial_prediction_and_backpropagates():
    model = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        heatmap_feature_dim=16,
        heatmap_topk=8,
        refinement_k=0,
        cascade_stages=1,
        cascade_attention_heads=4,
        cascade_radius_normalized=0.2,
    ).train()
    points = torch.randn(2, 64, 6, requires_grad=True)
    target = torch.randn(2, 85, 3) * 0.1
    details = model.forward_with_details(points)

    assert details["cascade_aux_predictions"].shape == (2, 1, 85, 3)
    assert details["cascade_aux_logits"].shape == (2, 1, 85, 64)
    assert details["cascade_stage_predictions"].shape == (2, 1, 85, 3)
    assert torch.equal(
        details["cascade_initial_prediction"],
        details["cascade_aux_predictions"][:, 0],
    )
    # The zero residual is an exact identity before the first optimizer step.
    assert torch.equal(details["coarse"], details["cascade_initial_prediction"])

    losses = proposal_landmark_loss(
        details["final"],
        target,
        torch.tensor([40.0, 40.0]),
        heatmap_logits=details["heatmap_logits"],
        heatmap_surface_points=details["surface_candidates"],
        heatmap_weight=0.1,
        cascade_aux_predictions=details["cascade_aux_predictions"],
        cascade_aux_logits=details["cascade_aux_logits"],
        cascade_coordinate_weight=0.25,
        cascade_heatmap_weight=0.05,
        cascade_heatmap_sigma_mm=2.0,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert points.grad is not None and torch.isfinite(points.grad).all()
    residual = model.cascade_layers[0].residual_query_projection
    assert residual.weight.grad is not None
    assert torch.isfinite(residual.weight.grad).all()


def test_two_stage_cascade_shapes_and_strict_checkpoint_reconstruction():
    config = {
        "backbone": "pointnext",
        "decoder": "surface_heatmap",
        "encoder_config": _tiny_encoder_config(),
        "four_heads": True,
        "heatmap_feature_dim": 16,
        "heatmap_topk": 8,
        "heatmap_coordinate_temperature": 1.0,
        "refinement_k": 0,
        "refinement_cap_normalized": 0.0,
        "refinement_anchor": "raw",
        "refinement_mode": "geometry-offset",
        "refinement_stages": 1,
        "refinement_hidden_dim": 32,
        "refinement_temperature": 1.0,
        "cascade_stages": 2,
        "cascade_attention_heads": 4,
        "cascade_radius_normalized": 0.2,
        "cascade_radius_decay": 0.5,
        "cascade_dropout": 0.0,
    }
    original = build_landmark_model(config).eval()
    reconstructed = build_landmark_model(config).eval()
    reconstructed.load_state_dict(original.state_dict(), strict=True)
    points = torch.randn(1, 64, 6)
    with torch.no_grad():
        details = reconstructed.forward_with_details(points)
    assert details["cascade_aux_predictions"].shape == (1, 2, 85, 3)
    assert details["cascade_stage_predictions"].shape == (1, 2, 85, 3)
    assert reconstructed.cascade_layers[0].radius_normalized == pytest.approx(0.2)
    assert reconstructed.cascade_layers[1].radius_normalized == pytest.approx(0.1)


def test_cascade_cli_is_explicit_and_serialized():
    args = build_parser().parse_args(
        [
            "fit-landmarks",
            "--folds-json", "folds.json",
            "--outer-fold", "0",
            "--predictions-json", "predictions.json",
            "--calibration-json", "calibration.json",
            "--output-dir", "runs/cascade",
            "--backbone", "pointnext",
            "--landmark-decoder", "surface-heatmap",
            "--heatmap-feature-dim", "256",
            "--heatmap-weight", "0.1",
            "--cascade-stages", "1",
            "--cascade-attention-heads", "8",
            "--cascade-radius-mm", "8",
            "--cascade-coordinate-weight", "0.25",
            "--cascade-heatmap-weight", "0.05",
            "--refinement-k", "32",
        ]
    )
    config = landmark_model_config(args, local_scale=40.0)
    losses = landmark_loss_config(args)
    assert config["cascade_stages"] == 1
    assert config["cascade_attention_heads"] == 8
    assert config["cascade_radius_normalized"] == pytest.approx(0.2)
    assert losses["cascade_coordinate"] == pytest.approx(0.25)
    assert losses["cascade_heatmap"] == pytest.approx(0.05)
    assert isinstance(build_landmark_model(config), PointNeXtSurfaceHeatmapRegressor)


def test_cascade_rejects_uncontrolled_combinations_and_missing_auxiliary_losses():
    base = [
        "fit-landmarks",
        "--folds-json", "folds.json",
        "--outer-fold", "0",
        "--predictions-json", "predictions.json",
        "--calibration-json", "calibration.json",
        "--output-dir", "runs/invalid",
        "--backbone", "pointnext",
        "--landmark-decoder", "surface-heatmap",
        "--heatmap-weight", "0.1",
        "--cascade-stages", "1",
    ]
    args = build_parser().parse_args(base)
    with pytest.raises(ValueError, match="requires positive"):
        landmark_loss_config(args)

    args = build_parser().parse_args(
        base
        + [
            "--cascade-coordinate-weight", "0.25",
            "--cascade-heatmap-weight", "0.05",
            "--bilateral-mode", "shared-latent",
        ]
    )
    with pytest.raises(ValueError, match="separate experiments"):
        landmark_model_config(args, local_scale=40.0)


def test_legacy_surface_heatmap_state_keys_remain_unchanged():
    legacy = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        heatmap_feature_dim=16,
        heatmap_topk=8,
        refinement_k=0,
    )
    explicit_zero = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        heatmap_feature_dim=16,
        heatmap_topk=8,
        refinement_k=0,
        cascade_stages=0,
    )
    assert tuple(legacy.state_dict()) == tuple(explicit_zero.state_dict())


def test_cascade_warm_start_accepts_only_new_cascade_state_keys(tmp_path):
    target_config = {
        "backbone": "pointnext",
        "decoder": "surface_heatmap",
        "encoder_config": _tiny_encoder_config(),
        "four_heads": True,
        "heatmap_feature_dim": 16,
        "heatmap_topk": 8,
        "heatmap_coordinate_temperature": 1.0,
        "refinement_k": 0,
        "refinement_cap_normalized": 0.0,
        "refinement_anchor": "raw",
        "refinement_mode": "geometry-offset",
        "refinement_stages": 1,
        "refinement_hidden_dim": 32,
        "refinement_temperature": 1.0,
        "cascade_stages": 1,
        "cascade_attention_heads": 4,
        "cascade_radius_normalized": 0.2,
        "cascade_radius_decay": 0.5,
        "cascade_dropout": 0.0,
    }
    cascade_keys = {
        "cascade_stages",
        "cascade_attention_heads",
        "cascade_radius_normalized",
        "cascade_radius_decay",
        "cascade_dropout",
    }
    source_config = {
        key: value for key, value in target_config.items() if key not in cascade_keys
    }
    source_model = build_landmark_model(source_config)
    losses = {
        "anchor": 0.0,
        "spacing": 0.01,
        "surface": 0.0,
        "heatmap": 0.1,
        "heatmap_sigma_mm": 2.0,
        "heatmap_distance": "euclidean",
        "cascade_coordinate": 0.25,
        "cascade_heatmap": 0.05,
    }
    data_config = {
        "outer_fold": 0,
        "train_ids": ["TRAIN"],
        "validation_ids": ["VAL"],
        "num_points": 64,
        "artifact_checksums": {
            "folds_json_sha256": "folds",
            "predictions_json_sha256": "predictions",
            "calibration_json_sha256": "calibration",
        },
    }
    source_data = {
        **data_config,
        "loss_weights": {
            key: value
            for key, value in losses.items()
            if not key.startswith("cascade_")
        },
    }
    path = tmp_path / "best_landmarks.pt"
    torch.save(
        {
            "component_schema_version": 1,
            "component": "landmarks",
            "model_config": source_config,
            "model_state_dict": source_model.state_dict(),
            "data_config": source_data,
            "epoch": 10,
            "metrics": {"best_md_mm": 1.25},
        },
        path,
    )
    target_model = build_landmark_model(target_config)
    report = _initialize_cascade_from_checkpoint(
        target_model,
        str(path),
        target_config,
        data_config,
        losses,
    )
    assert report["source_epoch"] == 10
    assert report["source_best_md_mm"] == pytest.approx(1.25)
    assert report["new_cascade_state_keys"]
    assert all(
        key.startswith("cascade_layers.")
        for key in report["new_cascade_state_keys"]
    )
