import numpy as np
import pytest
import torch
import trimesh

from src.geodesic import (
    compute_vertex_geodesics,
    sample_geodesic_distances_from_barycentric,
)
from src.geodesic_diagnostics import (
    candidate_retrieval_metrics,
    confounding_probability_mass,
)
from src.losses import proposal_landmark_loss, surface_geodesic_heatmap_kl
from src.pointnext_model import default_pointnext_config
from src.preprocessing import sample_mesh_surface, sample_mesh_surface_with_metadata
from src.proposal_models import PointNeXtSurfaceHeatmapRegressor, build_landmark_model
from src.surface import project_points_to_mesh_with_faces
from train_pipeline import build_parser, landmark_loss_config, landmark_model_config


def _two_close_disconnected_triangles():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
        ],
        dtype=np.float32,
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        process=False,
    )


def _tiny_encoder_config():
    return {
        **default_pointnext_config(width=8, variant="s"),
        "strides": [1, 2, 2, 2, 2],
        "nsample": 8,
    }


def test_detailed_sampling_is_bitwise_compatible_with_existing_sampler():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
    old = sample_mesh_surface(mesh, num_points=64, seed=17)
    detailed = sample_mesh_surface_with_metadata(mesh, num_points=64, seed=17)
    assert np.array_equal(old, detailed.features)
    assert detailed.face_indices.shape == (64,)
    assert detailed.barycentric.shape == (64, 3)
    assert np.allclose(detailed.barycentric.sum(axis=1), 1.0)


def test_mesh_geodesic_does_not_cross_disconnected_nearby_sheet():
    mesh = _two_close_disconnected_triangles()
    landmarks = np.repeat([[0.2, 0.2, 0.0]], 85, axis=0).astype(np.float32)
    cache = compute_vertex_geodesics(mesh, landmarks, "left")
    assert np.isfinite(cache["vertex_distances_mm"][:, :3]).all()
    assert np.isinf(cache["vertex_distances_mm"][:, 3:]).all()
    distances = sample_geodesic_distances_from_barycentric(
        mesh,
        np.asarray([0, 1]),
        np.asarray([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]], dtype=np.float32),
        cache,
    )
    assert np.isfinite(distances[:, 0]).all()
    assert np.isinf(distances[:, 1]).all()


def test_projection_reports_deterministic_source_face():
    mesh = _two_close_disconnected_triangles()
    projected, faces = project_points_to_mesh_with_faces(
        np.asarray([[0.2, 0.2, 0.02], [0.2, 0.2, 0.09]], dtype=np.float32),
        mesh,
    )
    assert faces.tolist() == [0, 1]
    assert projected.shape == (2, 3)


def test_geodesic_heatmap_kl_and_surface_vote_are_differentiable():
    model = PointNeXtSurfaceHeatmapRegressor(
        encoder_config=_tiny_encoder_config(),
        heatmap_feature_dim=16,
        heatmap_topk=8,
        surface_voting=True,
        vote_cap_normalized=0.2,
        vote_fusion_epsilon_normalized=0.01,
        refinement_k=0,
    ).train()
    points = torch.randn(1, 64, 6, requires_grad=True)
    target = points[:, :1, :3].expand(-1, 85, -1).contiguous()
    details = model.forward_with_details(points)
    geodesic = torch.cdist(target, details["surface_candidates"]) * 40.0
    assert details["surface_vote_offsets"].shape == (1, 85, 64, 3)
    assert details["coarse"].shape == (1, 85, 3)
    losses = proposal_landmark_loss(
        details["final"],
        target,
        torch.tensor([40.0]),
        heatmap_logits=details["heatmap_logits"],
        heatmap_surface_points=details["surface_candidates"],
        heatmap_geodesic_distances_mm=geodesic,
        heatmap_weight=0.1,
        heatmap_sigma_mm=2.0,
        vote_offsets=details["surface_vote_offsets"],
        vote_weight=0.1,
        vote_radius_mm=1000.0,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["heatmap"])
    assert torch.isfinite(losses["vote"])
    assert model.vote_query_projections[0].weight.grad is not None
    assert torch.isfinite(points.grad).all()


def test_geodesic_kl_prefers_correct_candidate_and_handles_infinity():
    distances = torch.full((1, 85, 2), float("inf"))
    distances[:, :, 0] = 0.0
    good = torch.tensor([8.0, -8.0]).reshape(1, 1, 2).expand(1, 85, 2)
    bad = -good
    good_loss = surface_geodesic_heatmap_kl(good, distances, sigma_mm=2.0)
    bad_loss = surface_geodesic_heatmap_kl(bad, distances, sigma_mm=2.0)
    assert torch.isfinite(good_loss)
    assert 0.0 <= good_loss < bad_loss


def test_geodesic_voting_cli_is_explicit_and_reconstructable():
    args = build_parser().parse_args(
        [
            "fit-landmarks",
            "--folds-json", "folds.json",
            "--outer-fold", "0",
            "--predictions-json", "predictions.json",
            "--calibration-json", "calibration.json",
            "--output-dir", "runs/geodesic_vote",
            "--backbone", "pointnext",
            "--landmark-decoder", "surface-heatmap",
            "--heatmap-weight", "0.1",
            "--heatmap-distance", "geodesic",
            "--geodesic-cache-dir", "artifacts/geodesic/fold0",
            "--surface-voting",
            "--vote-weight", "0.1",
            "--vote-radius-mm", "6",
            "--vote-cap-mm", "6",
            "--refinement-k", "32",
        ]
    )
    config = landmark_model_config(args, local_scale=40.0)
    weights = landmark_loss_config(args)
    assert config["surface_voting"] is True
    assert config["vote_cap_normalized"] == pytest.approx(0.15)
    assert weights["heatmap_distance"] == "geodesic"
    assert weights["vote"] == pytest.approx(0.1)
    original = build_landmark_model(config)
    reconstructed = build_landmark_model(config)
    reconstructed.load_state_dict(original.state_dict(), strict=True)


def test_candidate_and_confounding_diagnostics_have_expected_values():
    logits = np.zeros((85, 3), dtype=np.float32)
    logits[:, 1] = 3.0
    candidates = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    targets = np.repeat([[1, 0, 0]], 85, axis=0).astype(np.float32)
    ranks, oracle = candidate_retrieval_metrics(logits, candidates, targets, 1.0, [1, 2])
    assert np.all(ranks == 1)
    assert np.all(oracle[1] == 0.0)

    geodesic = np.zeros((85, 3), dtype=np.float32)
    geodesic[:, 1] = 10.0
    normals = np.asarray([[0, 0, 1], [0, 0, -1], [0, 0, 1]], dtype=np.float32)
    source_normals = np.repeat([[0, 0, 1]], 85, axis=0).astype(np.float32)
    shortcut, opposite = confounding_probability_mass(
        logits, candidates, targets, 1.0, geodesic, normals, source_normals,
        euclidean_close_mm=0.5, geodesic_far_mm=8.0, normal_dot_threshold=0.0,
    )
    assert np.all(shortcut > 0.8)
    assert np.allclose(shortcut, opposite)
