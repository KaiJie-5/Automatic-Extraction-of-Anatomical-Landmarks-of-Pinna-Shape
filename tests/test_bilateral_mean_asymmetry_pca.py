import json

import numpy as np
import pytest

from src.estimator import LandmarkExtractor
from src.shape_prior.bilateral_pca import (
    BilateralMeanAsymmetryPCAPrior,
    gate_bilateral_contours,
    normalise_contour_gate,
)
from src.shape_prior.pca import PCAShapePrior
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


def test_contour_gate_uses_bilateral_only_for_selected_ranges():
    independent = np.zeros((2, 85, 3), dtype=np.float32)
    bilateral = np.ones((2, 85, 3), dtype=np.float32)
    actual = gate_bilateral_contours(
        independent,
        bilateral,
        ["superior_antihelix", "concha"],
    )
    np.testing.assert_array_equal(actual[:, :25], np.zeros_like(actual[:, :25]))
    np.testing.assert_array_equal(
        actual[:, 25:55], np.ones_like(actual[:, 25:55])
    )
    np.testing.assert_array_equal(
        actual[:, 55:75], np.zeros_like(actual[:, 55:75])
    )
    np.testing.assert_array_equal(actual[:, 75:], np.ones_like(actual[:, 75:]))
    assert normalise_contour_gate(
        ["superior_antihelix", "concha"]
    ) == ("concha", "superior_antihelix")
    with pytest.raises(ValueError, match="duplicates"):
        normalise_contour_gate(["concha", "concha"])


def test_estimator_pair_postprocess_preserves_independent_contours():
    pairs = _pairs()
    independent_prior = PCAShapePrior.fit(
        pairs.reshape(-1, 85, 3), n_components=4, beta=0.5
    )
    bilateral_prior = BilateralMeanAsymmetryPCAPrior.fit(
        pairs,
        common_components=5,
        asymmetry_components=3,
        common_beta=0.625,
        asymmetry_beta=0.5,
    )
    extractor = LandmarkExtractor.__new__(LandmarkExtractor)
    extractor.pca_shape_prior = independent_prior
    extractor.bilateral_pca_shape_prior = bilateral_prior
    extractor.bilateral_pca_contour_gate = (
        "concha",
        "superior_antihelix",
    )
    independent = independent_prior.blend(pairs[0])
    bilateral = bilateral_prior.blend_pair(pairs[0])
    expected = gate_bilateral_contours(
        independent,
        bilateral,
        extractor.bilateral_pca_contour_gate,
    )
    actual = extractor._apply_pca_postprocess_pair(pairs[0])
    np.testing.assert_array_equal(actual, expected)


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
            "--independent-prior-path",
            "independent.npz",
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
            "--contour-gate",
            "concha",
            "superior_antihelix",
            "--skip-projection",
            "--output",
            "screen.json",
        ]
    )
    assert args.common_components == [16, 32]
    assert args.asymmetry_components == [4, 8]
    assert args.common_betas == [0.25, 0.5]
    assert args.asymmetry_betas == [0.5, 0.75]
    assert args.independent_prior_path == "independent.npz"
    assert args.contour_gate == ["concha", "superior_antihelix"]
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
                "contour_gate": ["concha", "superior_antihelix"],
                "independent_components": 32,
                "independent_beta": 0.5,
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
    assert summary["contour_gate"] == ["concha", "superior_antihelix"]
    assert summary["postprocess_mode"] == (
        "contour_gated_bilateral_over_independent_pca"
    )
    assert summary["independent_components"] == 32
    assert summary["independent_beta"] == 0.5
    assert summary["promotion_rule"]["promoted"]
