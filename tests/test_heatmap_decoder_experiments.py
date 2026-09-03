import numpy as np
import pytest
import torch

from src.heatmap_diagnostics import _configuration_key, _pearson, heatmap_uncertainty
from src.pointnext_model import default_pointnext_config
from src.proposal_models import (
    PointNeXtSurfaceHeatmapRegressor,
    build_landmark_model,
)
from train_pipeline import build_parser, landmark_model_config


def _tiny_encoder_config():
    return {
        **default_pointnext_config(width=8, variant="s"),
        "strides": [1, 2, 2, 2, 2],
        "nsample": 8,
    }


def test_decode_overrides_reproduce_saved_defaults():
    model = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        heatmap_feature_dim=16,
        heatmap_topk=8,
        heatmap_coordinate_temperature=0.5,
        refinement_k=0,
    ).eval()
    points = torch.randn(1, 64, 6)
    with torch.no_grad():
        details = model.forward_with_details(points)
        decoded = model.decode_surface_coordinates(
            details["heatmap_logits"],
            details["surface_candidates"],
            topk=8,
            temperature=0.5,
        )
    assert torch.equal(decoded, details["coarse"])
    assert details["decoded_point_features"].shape == (1, 64, 16)
    assert details["landmark_query_features"].shape == (85, 16)


def test_feature_attention_refiner_is_two_stage_and_differentiable():
    model = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        heatmap_feature_dim=16,
        heatmap_topk=8,
        refinement_k=32,
        refinement_mode="feature-attention",
        refinement_stages=2,
        refinement_hidden_dim=32,
        refinement_temperature=0.5,
    ).train()
    points = torch.randn(1, 64, 6, requires_grad=True)
    details = model.forward_with_details(points)
    assert details["coarse"].shape == (1, 85, 3)
    assert details["final"].shape == (1, 85, 3)
    assert details["refinement_stage_predictions"].shape == (1, 2, 85, 3)
    assert torch.isfinite(details["final"]).all()
    details["final"].square().mean().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert model.refiner.stages[0].score_head[-1].weight.grad is not None


def test_legacy_heatmap_config_still_strictly_reconstructs():
    config = {
        "backbone": "pointnext",
        "decoder": "surface_heatmap",
        "encoder_config": _tiny_encoder_config(),
        "four_heads": True,
        "heatmap_feature_dim": 16,
        "heatmap_topk": 8,
        "heatmap_coordinate_temperature": 1.0,
        "refinement_k": 32,
        "refinement_cap_normalized": 0.125,
        "refinement_anchor": "raw",
    }
    original = build_landmark_model(config)
    reconstructed = build_landmark_model(config)
    reconstructed.load_state_dict(original.state_dict(), strict=True)
    assert reconstructed.refinement_mode == "geometry-offset"
    assert not any("FeatureAware" in type(module).__name__ for module in reconstructed.modules())


def test_feature_attention_cli_is_explicit_and_checkpointed():
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
            "runs/feature_refiner",
            "--backbone",
            "pointnext",
            "--landmark-decoder",
            "surface-heatmap",
            "--heatmap-weight",
            "0.1",
            "--refinement-k",
            "32",
            "--refinement-mode",
            "feature-attention",
            "--refinement-stages",
            "2",
            "--refinement-hidden-dim",
            "128",
            "--refinement-temperature",
            "0.5",
        ]
    )
    config = landmark_model_config(args, local_scale=40.0)
    assert config["refinement_mode"] == "feature-attention"
    assert config["refinement_stages"] == 2
    assert config["refinement_hidden_dim"] == 128
    assert config["refinement_temperature"] == pytest.approx(0.5)
    assert isinstance(build_landmark_model(config), PointNeXtSurfaceHeatmapRegressor)


def test_feature_attention_rejects_missing_neighbourhood():
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
            "--backbone",
            "pointnext",
            "--landmark-decoder",
            "surface-heatmap",
            "--heatmap-weight",
            "0.1",
            "--refinement-mode",
            "feature-attention",
            "--refinement-stages",
            "2",
        ]
    )
    with pytest.raises(ValueError, match="requires --refinement-k"):
        landmark_model_config(args, local_scale=40.0)


def test_heatmap_diagnostic_cli_and_statistics_helpers():
    args = build_parser().parse_args(
        [
            "analyze-heatmap-decoder",
            "--checkpoint-path",
            "best_landmarks.pt",
            "--prior-path",
            "prior.npz",
            "--prior-manifest",
            "manifest.json",
            "--folds-json",
            "folds.json",
            "--predictions-json",
            "predictions.json",
            "--calibration-json",
            "calibration.json",
            "--top-k",
            "1",
            "8",
            "64",
            "--temperatures",
            "0.25",
            "0.5",
            "1.0",
            "--output",
            "diagnostic.json",
            "--projection-workers",
            "4",
        ]
    )
    assert args.top_k == [1, 8, 64]
    assert args.temperatures == [0.25, 0.5, 1.0]
    assert args.projection_workers == 4
    assert _configuration_key(64, 0.5) == "topk_64_temperature_0p5"
    assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert _pearson([1, 1], [2, 3]) is None

    logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]])
    uncertainty = heatmap_uncertainty(logits)
    for value in uncertainty.values():
        assert value.shape == (1, 2)
        assert torch.isfinite(value).all()
    assert np.all(uncertainty["entropy"].detach().numpy() >= 0.0)
    assert np.all(uncertainty["entropy"].detach().numpy() <= 1.0)
