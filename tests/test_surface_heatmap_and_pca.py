import json

import numpy as np
import pytest
import torch

from src.losses import proposal_landmark_loss, surface_heatmap_kl
from src.pointnext_model import PointNeXtEncoder, default_pointnext_config
from src.proposal_models import (
    PointNeXtSurfaceHeatmapRegressor,
    build_landmark_model,
)
from src.shape_prior.pca import PCAShapePrior
from src.shape_prior.summarize_prior import summarize
from train_pipeline import build_parser, landmark_loss_config, landmark_model_config


def _tiny_encoder_config():
    return {
        **default_pointnext_config(width=8, variant="s"),
        "strides": [1, 2, 2, 2, 2],
        "nsample": 8,
    }


def test_pointnext_forward_features_preserves_legacy_output_and_state_keys():
    encoder = PointNeXtEncoder(**_tiny_encoder_config()).eval()
    keys_before = tuple(encoder.state_dict())
    points = torch.randn(1, 64, 6)
    with torch.no_grad():
        details = encoder.forward_features(points)
        legacy = encoder(points)
    assert tuple(encoder.state_dict()) == keys_before
    assert len(details["xyz"]) == len(details["features"]) == 5
    assert torch.equal(details["global"], legacy)


def test_surface_heatmap_decoder_forward_details_and_backward():
    model = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        four_heads=True,
        heatmap_feature_dim=16,
        heatmap_topk=8,
        refinement_k=0,
    ).train()
    points = torch.randn(1, 64, 6, requires_grad=True)
    target = torch.randn(1, 85, 3) * 0.1
    details = model.forward_with_details(points)
    assert details["coarse"].shape == (1, 85, 3)
    assert details["final"].shape == (1, 85, 3)
    assert details["heatmap_logits"].shape == (1, 85, 64)
    assert details["surface_candidates"].shape == (1, 64, 3)
    assert torch.equal(details["coarse"], details["final"])
    assert torch.equal(model(points), model.forward_with_details(points)["final"])

    losses = proposal_landmark_loss(
        details["final"],
        target,
        torch.tensor([40.0]),
        spacing_weight=0.01,
        heatmap_logits=details["heatmap_logits"],
        heatmap_surface_points=details["surface_candidates"],
        heatmap_weight=0.1,
        heatmap_sigma_mm=2.0,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert points.grad is not None and torch.isfinite(points.grad).all()
    assert model.query_embeddings[0].grad is not None


def test_surface_heatmap_kl_prefers_correct_surface_candidate():
    candidates = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    target = torch.zeros(1, 85, 3)
    good = torch.tensor([10.0, -10.0]).view(1, 1, 2).expand(1, 85, 2)
    bad = torch.tensor([-10.0, 10.0]).view(1, 1, 2).expand(1, 85, 2)
    good_loss = surface_heatmap_kl(good, candidates, target, 1.0, 1.0)
    bad_loss = surface_heatmap_kl(bad, candidates, target, 1.0, 1.0)
    assert good_loss < bad_loss
    assert good_loss >= 0.0


def test_surface_heatmap_cli_is_explicit_and_checkpoint_reconstructable():
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
            "runs/heatmap",
            "--backbone",
            "pointnext",
            "--landmark-decoder",
            "surface-heatmap",
            "--heatmap-weight",
            "0.1",
            "--refinement-k",
            "32",
        ]
    )
    config = landmark_model_config(args, local_scale=40.0)
    weights = landmark_loss_config(args)
    assert config["decoder"] == "surface_heatmap"
    assert config["backbone"] == "pointnext"
    assert config["refinement_k"] == 32
    assert weights["heatmap"] == pytest.approx(0.1)
    assert isinstance(build_landmark_model(config), PointNeXtSurfaceHeatmapRegressor)

    invalid = build_parser().parse_args(
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
            "pointnet2",
            "--landmark-decoder",
            "surface-heatmap",
            "--heatmap-weight",
            "0.1",
        ]
    )
    with pytest.raises(ValueError, match="requires --backbone pointnext"):
        landmark_model_config(invalid, local_scale=40.0)


def test_pca_blend_rejects_extrapolation_and_component_zero():
    rng = np.random.default_rng(42)
    prior = PCAShapePrior.fit(rng.normal(size=(8, 85, 3)).astype(np.float32), 4)
    prediction = rng.normal(size=(85, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="beta"):
        prior.blend(prediction, beta=1.1)
    with pytest.raises(ValueError, match="settings"):
        PCAShapePrior(prior.mean_shape, prior.components, 0)


def test_pca_confirmation_summary_enforces_pooled_and_three_fold_rule(tmp_path):
    root = tmp_path / "reports"
    root.mkdir()
    for fold in range(5):
        for seed in (42, 43, 44):
            baseline = 2.0 + seed * 0.0001
            candidate = baseline - 0.1 if fold < 3 else baseline + 0.02
            ear_row = {
                "projected_md_mm": baseline,
                "pca_projected_md_mm": candidate,
            }
            report = {
                "schema_version": 2,
                "component": "fold_projection_aware_pca_evaluation",
                "outer_fold": fold,
                "run_seed": seed,
                "components": 32,
                "beta": 0.5,
                "ear_count": 2,
                "projected_mean_md_mm": baseline,
                "pca_projected_mean_md_mm": candidate,
                "subject_level_raw_and_pca_errors": {
                    f"S{fold}": {"left": ear_row, "right": ear_row}
                },
            }
            (root / f"fold{fold}_seed{seed}.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

    summary = summarize(root, (42, 43, 44))
    assert summary["report_count"] == 15
    assert summary["improved_folds"] == 3
    assert summary["pooled_improvement_mm"] > 0.0
    assert summary["promotion_rule"]["promoted"]
