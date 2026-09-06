import json

import numpy as np
import pytest

from src.shape_prior.bilateral_pca import BilateralMeanAsymmetryPCAPrior
from src.shape_prior.summarize_bilateral_prior import summarize
from train_pipeline import build_parser


def _pairs(seed=42, count=12):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(count, 85, 3)).astype(np.float32)
    asymmetry = (
        0.15 * rng.normal(size=(count, 85, 3))
    ).astype(np.float32)
    return np.stack([common + asymmetry, common - asymmetry], axis=1)


def test_bilateral_prior_zero_blend_preserves_both_ears():
    pairs = _pairs()
    prior = BilateralMeanAsymmetryPCAPrior.fit(
        pairs, common_components=8, asymmetry_components=8
    )
    actual = prior.blend_pair(
        pairs[0], common_beta=0.0, asymmetry_beta=0.0
    )
    np.testing.assert_allclose(actual, pairs[0], rtol=1e-6, atol=1e-6)


def test_bilateral_prior_full_training_span_reconstructs_training_pair():
    pairs = _pairs(count=8)
    prior = BilateralMeanAsymmetryPCAPrior.fit(
        pairs, common_components=8, asymmetry_components=8
    )
    actual = prior.blend_pair(
        pairs[3], common_beta=1.0, asymmetry_beta=1.0
    )
    np.testing.assert_allclose(actual, pairs[3], rtol=2e-5, atol=2e-5)


def test_bilateral_prior_save_load_and_batched_blend(tmp_path):
    pairs = _pairs()
    prior = BilateralMeanAsymmetryPCAPrior.fit(
        pairs,
        common_components=7,
        asymmetry_components=5,
        common_beta=0.5,
        asymmetry_beta=0.75,
    )
    path = tmp_path / "bilateral_prior.npz"
    prior.save(path)
    loaded = BilateralMeanAsymmetryPCAPrior.load(path)
    expected = prior.blend(pairs[:2])
    actual = loaded.blend(pairs[:2])
    assert actual.shape == (2, 2, 85, 3)
    np.testing.assert_array_equal(actual, expected)


def test_bilateral_prior_rejects_invalid_pair_and_overlarge_components():
    pairs = _pairs()
    prior = BilateralMeanAsymmetryPCAPrior.fit(
        pairs, common_components=4, asymmetry_components=4
    )
    with pytest.raises(ValueError, match="shape"):
        prior.blend_pair(np.zeros((85, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="common_components"):
        prior.blend_pair(pairs[0], common_components=5)


def test_bilateral_pca_cli_grid_is_explicit():
    args = build_parser().parse_args(
        [
            "evaluate-bilateral-pca-prior",
            "--checkpoint-path",
            "model.pt",
            "--prior-path",
            "prior.npz",
            "--prior-manifest",
            "manifest.json",
            "--reference-report",
            "reference.json",
            "--folds-json",
            "folds.json",
            "--predictions-json",
            "centres.json",
            "--calibration-json",
            "calibration.json",
            "--common-components",
            "16",
            "32",
            "--asymmetry-components",
            "4",
            "8",
            "--common-betas",
            "0.25",
            "0.5",
            "--asymmetry-betas",
            "0.5",
            "0.75",
            "--skip-projection",
            "--output",
            "screen.json",
        ]
    )
    assert args.common_components == [16, 32]
    assert args.asymmetry_components == [4, 8]
    assert args.common_betas == [0.25, 0.5]
    assert args.asymmetry_betas == [0.5, 0.75]
    assert args.skip_projection


def test_bilateral_confirmation_compares_against_independent_prior(tmp_path):
    root = tmp_path / "reports"
    root.mkdir()
    setting = {
        "common_components": 32,
        "asymmetry_components": 8,
        "common_beta": 0.5,
        "asymmetry_beta": 0.75,
    }
    for fold in range(5):
        for seed in (42, 43, 44):
            reference = 1.3 + seed * 0.0001
            candidate = reference - 0.02 if fold < 3 else reference + 0.005
            row = {
                "reference_pca_projected_md_mm": reference,
                "pca_projected_md_mm": candidate,
            }
            report = {
                "schema_version": 1,
                "component": "fold_bilateral_mean_asymmetry_pca_evaluation",
                "projection_evaluated": True,
                "outer_fold": fold,
                "run_seed": seed,
                "candidate_count": 1,
                "selected_setting": setting,
                "ear_count": 2,
                "reference_mean_md_mm": reference,
                "pca_projected_mean_md_mm": candidate,
                "subject_level_raw_and_pca_errors": {
                    f"S{fold}": {"left": row, "right": row}
                },
            }
            (root / f"fold{fold}_seed{seed}.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
    summary = summarize(root, (42, 43, 44))
    assert summary["improved_folds"] == 3
    assert summary["pooled_improvement_mm"] > 0.0
    assert summary["promotion_rule"]["promoted"]
