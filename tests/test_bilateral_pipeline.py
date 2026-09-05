import pytest
import torch

from src.pointnext_model import default_pointnext_config
from src.proposal_models import (
    BilateralPointNeXtSurfaceHeatmapRegressor,
    PointNeXtSurfaceHeatmapRegressor,
    build_landmark_model,
)
from src.training import _flatten_landmark_batch
from train_pipeline import build_parser, landmark_model_config


def _tiny_encoder_config():
    return {
        **default_pointnext_config(width=8, variant="s"),
        "strides": [1, 2, 2, 2, 2],
        "nsample": 8,
    }


def _model_config(mode):
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
    }
    if mode != "none":
        config.update(
            bilateral_mode=mode,
            bilateral_attention_heads=4,
            bilateral_attention_layers=1,
            bilateral_dropout=0.0,
        )
    return config


@pytest.mark.parametrize(
    "mode", ("shared-latent", "landmark-cross-attention")
)
def test_bilateral_models_have_paired_shapes_and_backward(mode):
    model = build_landmark_model(_model_config(mode)).train()
    assert isinstance(model, BilateralPointNeXtSurfaceHeatmapRegressor)
    points = torch.randn(2, 2, 64, 6, requires_grad=True)
    details = model.forward_with_details(points)
    assert details["coarse"].shape == (2, 2, 85, 3)
    assert details["final"].shape == (2, 2, 85, 3)
    assert details["heatmap_logits"].shape == (2, 2, 85, 64)
    assert details["surface_candidates"].shape == (2, 2, 64, 3)
    assert details["landmark_query_features"].shape == (2, 2, 85, 16)
    details["final"].square().mean().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()


@pytest.mark.parametrize(
    "mode", ("shared-latent", "landmark-cross-attention")
)
def test_bilateral_fusion_is_ear_swap_equivariant(mode):
    torch.manual_seed(42)
    model = build_landmark_model(_model_config(mode)).eval()
    points = torch.randn(1, 2, 64, 6)
    with torch.no_grad():
        original = model(points)
        swapped = model(points.flip(1)).flip(1)
    torch.testing.assert_close(original, swapped, rtol=1e-5, atol=1e-6)


def test_bilateral_checkpoint_reconstructs_strictly_and_legacy_is_unchanged():
    config = _model_config("landmark-cross-attention")
    model = build_landmark_model(config)
    reconstructed = build_landmark_model(config)
    reconstructed.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(reconstructed, BilateralPointNeXtSurfaceHeatmapRegressor)
    assert isinstance(
        build_landmark_model(_model_config("none")),
        PointNeXtSurfaceHeatmapRegressor,
    )


def test_bilateral_training_batch_flattens_to_unchanged_per_ear_contract():
    prediction = torch.randn(3, 2, 85, 3)
    target = torch.randn(3, 2, 85, 3)
    scale = torch.full((3, 2), 40.0)
    logits = torch.randn(3, 2, 85, 64)
    candidates = torch.randn(3, 2, 64, 3)
    values = _flatten_landmark_batch(
        prediction, target, scale, None, logits, candidates
    )
    flat_prediction, flat_target, flat_scale, _, flat_logits, flat_points, count = values
    assert flat_prediction.shape == flat_target.shape == (6, 85, 3)
    assert flat_scale.shape == (6,)
    assert flat_logits.shape == (6, 85, 64)
    assert flat_points.shape == (6, 64, 3)
    assert count == 6


@pytest.mark.parametrize(
    "mode", ("shared-latent", "landmark-cross-attention")
)
def test_bilateral_cli_serializes_reconstructable_configuration(mode):
    args = build_parser().parse_args(
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
            "runs/bilateral",
            "--backbone",
            "pointnext",
            "--landmark-decoder",
            "surface-heatmap",
            "--heatmap-feature-dim",
            "256",
            "--heatmap-weight",
            "0.1",
            "--bilateral-mode",
            mode,
            "--bilateral-attention-heads",
            "8",
        ]
    )
    config = landmark_model_config(args, local_scale=40.0)
    assert config["bilateral_mode"] == mode
    assert config["bilateral_attention_heads"] == 8
    assert isinstance(
        build_landmark_model(config),
        BilateralPointNeXtSurfaceHeatmapRegressor,
    )


def test_bilateral_cli_rejects_non_heatmap_backbone():
    args = build_parser().parse_args(
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
            "--bilateral-mode",
            "shared-latent",
        ]
    )
    with pytest.raises(ValueError, match="requires --backbone pointnext"):
        landmark_model_config(args, local_scale=40.0)
