import numpy as np
import pytest
import torch
import trimesh

from src.curve import (
    CONTOUR_ANCHORS,
    CONTOUR_RANGES,
    contour_anchor_manifest,
    decode_connected_curve_paths,
    landmark_arc_fractions,
    median_landmark_arc_fractions,
)
from src.losses import surface_curve_arc_loss, surface_curve_field_kl
from src.pointnext_model import default_pointnext_config
from src.proposal_models import PointNeXtSurfaceHeatmapRegressor, build_landmark_model
from train_pipeline import build_parser, landmark_loss_config, landmark_model_config


def _fractions():
    values = np.empty(85, dtype=np.float32)
    for start, end in CONTOUR_RANGES:
        values[start:end] = np.linspace(0.0, 1.0, end - start)
    return values


def _landmarks(scale=1.0):
    result = np.zeros((85, 3), dtype=np.float32)
    for contour, (start, end) in enumerate(CONTOUR_RANGES):
        result[start:end, 0] = np.linspace(0.0, scale * (contour + 1), end - start)
        result[start:end, 1] = contour
    return result


def _tiny_encoder_config():
    return {
        **default_pointnext_config(width=8, variant="s"),
        "strides": [1, 2, 2, 2, 2],
        "nsample": 8,
    }


def test_annotation_arc_fractions_are_invariant_and_fold_median_is_ordered():
    first = landmark_arc_fractions(_landmarks())
    transformed = landmark_arc_fractions(_landmarks(scale=3.0) + 17.0)
    assert np.allclose(first, transformed)
    median = median_landmark_arc_fractions([_landmarks(), _landmarks(scale=2.0)])
    for start, end in CONTOUR_RANGES:
        assert median[start] == 0.0
        assert median[end - 1] == 1.0
        assert np.all(np.diff(median[start:end]) > 0.0)


def test_curve_field_and_arc_losses_are_differentiable_and_prefer_targets():
    points = 85
    distances = torch.full((1, 85, points), 20.0)
    indices = torch.arange(85)
    distances[0, indices, indices] = 0.0
    fractions = torch.from_numpy(_fractions()).unsqueeze(0)

    good_curve = torch.full((1, 4, points), -8.0, requires_grad=True)
    good_arc = torch.zeros(1, 4, points, requires_grad=True)
    with torch.no_grad():
        for contour, (start, end) in enumerate(CONTOUR_RANGES):
            good_curve[0, contour, start:end] = 8.0
            good_arc[0, contour, start:end] = fractions[0, start:end]
    bad_curve = -good_curve.detach()
    bad_arc = 1.0 - good_arc.detach()
    good_field_loss = surface_curve_field_kl(good_curve, distances, 2.0)
    bad_field_loss = surface_curve_field_kl(bad_curve, distances, 2.0)
    good_arc_loss = surface_curve_arc_loss(
        good_arc, distances, fractions, 2.0, 4.0
    )
    bad_arc_loss = surface_curve_arc_loss(
        bad_arc, distances, fractions, 2.0, 4.0
    )
    assert 0.0 <= good_field_loss < bad_field_loss
    assert 0.0 <= good_arc_loss < bad_arc_loss
    (good_field_loss + good_arc_loss).backward()
    assert torch.isfinite(good_curve.grad).all()
    assert torch.isfinite(good_arc.grad).all()


def test_surface_curve_model_outputs_shared_fields_and_strictly_reloads():
    model = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        heatmap_feature_dim=16,
        heatmap_topk=8,
        curve_enabled=True,
        curve_landmark_fractions=_fractions(),
        refinement_k=0,
    ).eval()
    points = torch.randn(1, 64, 6)
    with torch.no_grad():
        details = model.forward_with_details(points)
    assert details["final"].shape == (1, 85, 3)
    assert details["curve_logits"].shape == (1, 4, 64)
    assert details["curve_arc_coordinates"].shape == (1, 4, 64)
    assert torch.all((details["curve_arc_coordinates"] >= 0.0) & (details["curve_arc_coordinates"] <= 1.0))
    assert not torch.equal(details["unary_heatmap_logits"], details["heatmap_logits"])

    config = {
        "backbone": "pointnext",
        "decoder": "surface_curve",
        "encoder_config": _tiny_encoder_config(),
        "four_heads": True,
        "heatmap_feature_dim": 16,
        "heatmap_topk": 8,
        "heatmap_coordinate_temperature": 1.0,
        "curve_landmark_fractions": _fractions().tolist(),
        "curve_logit_weight": 0.5,
        "curve_arc_logit_weight": 0.25,
        "curve_arc_temperature": 0.1,
        "refinement_k": 0,
        "refinement_cap_normalized": 0.0,
        "refinement_anchor": "raw",
        "refinement_mode": "geometry-offset",
        "refinement_stages": 1,
        "refinement_hidden_dim": 128,
        "refinement_temperature": 1.0,
        "surface_voting": False,
        "vote_cap_normalized": 0.0,
        "vote_fusion_iterations": 3,
        "vote_fusion_epsilon_normalized": 0.0,
    }
    reconstructed = build_landmark_model(config)
    reconstructed.load_state_dict(model.state_dict(), strict=True)


class _IdentityTransform:
    @staticmethod
    def normalize_xyz(values):
        return np.asarray(values, dtype=np.float64)


def test_connected_curve_decoder_returns_ordered_mesh_edge_paths():
    vertices = np.asarray(
        [[x, y, 0.0] for x in range(101) for y in (0.0, 1.0)],
        dtype=np.float64,
    )
    faces = []
    for x in range(100):
        a, b = x * 2, x * 2 + 1
        c, d = (x + 1) * 2, (x + 1) * 2 + 1
        faces.extend([[a, c, b], [b, c, d]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    sample_xyz = vertices.copy()
    curve_logits = np.tile(np.where(vertices[:, 1] == 0.0, 8.0, -8.0), (4, 1))
    curve_arc = np.tile((vertices[:, 0] / 100.0), (4, 1))
    initial = np.zeros((85, 3), dtype=np.float32)
    for start, end in CONTOUR_RANGES:
        initial[start:end, 0] = _fractions()[start:end] * 100.0
    decoded, diagnostics = decode_connected_curve_paths(
        mesh,
        "left",
        _IdentityTransform(),
        sample_xyz,
        curve_logits,
        curve_arc,
        initial,
        _fractions(),
    )
    assert decoded.shape == (85, 3)
    for start, end in CONTOUR_RANGES:
        assert np.allclose(decoded[start:end, 1:], 0.0)
        assert np.all(np.diff(decoded[start:end, 0]) >= 0.0)
    assert all(not item["fallback"] for item in diagnostics.values())
    assert contour_anchor_manifest() == {
        "outer_helix": [0, 6, 22, 24],
        "concha": [25, 33, 42, 46, 50, 54],
        "inner_helix": [55, 64, 74],
        "superior_antihelix": [75, 84],
    }
    for name, anchors in zip(contour_anchor_manifest(), CONTOUR_ANCHORS):
        record = diagnostics[name]
        assert record["anchor_indices"] == list(anchors)
        assert record["section_count"] == len(anchors) - 1
        assert record["successful_section_count"] == len(anchors) - 1
        assert len(record["sections"]) == len(anchors) - 1


def test_surface_curve_cli_requires_explicit_geodesic_losses_and_reconstructs():
    args = build_parser().parse_args(
        [
            "fit-landmarks",
            "--folds-json", "folds.json",
            "--outer-fold", "0",
            "--predictions-json", "predictions.json",
            "--calibration-json", "calibration.json",
            "--output-dir", "runs/curve",
            "--backbone", "pointnext",
            "--landmark-decoder", "surface-curve",
            "--heatmap-weight", "0.1",
            "--heatmap-distance", "geodesic",
            "--geodesic-cache-dir", "artifacts/geodesic/fold0",
            "--curve-weight", "0.1",
            "--curve-arc-weight", "0.1",
            "--refinement-k", "32",
        ]
    )
    config = landmark_model_config(args, 40.0, _fractions())
    weights = landmark_loss_config(args)
    assert config["decoder"] == "surface_curve"
    assert weights["curve"] == pytest.approx(0.1)
    assert weights["curve_arc"] == pytest.approx(0.1)
    assert isinstance(build_landmark_model(config), PointNeXtSurfaceHeatmapRegressor)

    args.heatmap_distance = "euclidean"
    with pytest.raises(ValueError, match="requires --heatmap-distance geodesic"):
        landmark_loss_config(args)
