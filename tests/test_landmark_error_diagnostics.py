from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import trimesh

from src.landmark_error_diagnostics import (
    anatomical_frames, component_summary, decompose_errors, _project_with_workers,
)
from src.surface import project_points_to_mesh_with_faces


def _straight_contours():
    # Large jumps between contours must not influence endpoint tangents.
    target = np.zeros((85, 3))
    for row, (start, end) in enumerate(((0, 25), (25, 55), (55, 75), (75, 85))):
        target[start:end, 0] = np.arange(end - start)
        target[start:end, 1] = 100 * row
    return target, np.tile([0.0, 0.0, 1.0], (85, 1))


def test_frames_do_not_join_contours_and_partition_known_errors():
    target, normal = _straight_contours()
    basis, valid, _ = anatomical_frames(target, normal, np.zeros(85), "left")
    assert valid.all()
    np.testing.assert_allclose(basis, np.tile(np.eye(3), (85, 1, 1)))
    errors, signed, residual = decompose_errors(target + [3, -4, 12], target, basis, valid, "left")
    np.testing.assert_allclose(errors, 13)
    np.testing.assert_allclose(signed, np.tile([3, -4, 12], (85, 1)))
    assert residual < 1e-8
    summary = component_summary(signed, valid)
    assert summary["components"]["along_contour"]["squared_error_fraction"] == pytest.approx(9 / 169)
    assert summary["components"]["across_contour"]["mean_signed_mm"] == pytest.approx(-4)


def test_reflected_ears_have_same_canonical_signed_components():
    target, normal = _straight_contours()
    # Tilt the geometry to exercise every reflected coordinate.
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    target, normal = target @ rotation, normal @ rotation
    prediction = target + np.array([3, -4, 12]) @ rotation
    reflection = np.array([1, -1, 1])
    results = []
    for ear, scale in (("left", np.ones(3)), ("right", reflection)):
        basis, valid, _ = anatomical_frames(target * scale, normal * scale, np.zeros(85), ear)
        results.append(decompose_errors(prediction * scale, target * scale, basis, valid, ear)[1])
    np.testing.assert_allclose(results[0], results[1])


def test_invalid_frames_do_not_remove_landmarks_from_official_md():
    target, normal = _straight_contours()
    normal[0] = [0, 0, 0]
    normal[1] = [1, 0, 0]  # Chord orthogonal to the available tangent plane.
    distance = np.zeros(85)
    distance[2] = 0.6
    basis, valid, quality = anatomical_frames(target, normal, distance, "left", 0.5)
    assert not valid[:3].any()
    assert valid[3:].all()
    assert quality["degenerate_geometry"][:2].all()
    assert quality["target_far_from_crop_surface"][2]
    assert np.isnan(basis[:3]).all()
    error, signed, _ = decompose_errors(target + [0, 0, 2], target, basis, valid, "left")
    assert error.shape == (85,)
    assert error.mean() == pytest.approx(2)
    assert component_summary(signed, valid)["valid_landmarks"] == 82
    assert component_summary(signed, np.zeros(85, dtype=bool))["components"] is None


def test_parallel_projection_preserves_point_order_and_source_faces():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 3], [2, 0, 3], [0, 2, 3]],
        faces=[[0, 1, 2], [3, 4, 5]], process=False,
    )
    points = np.array([[0.2, 0.3, 2.9], [0.2, 0.3, -0.2], [0.7, 0.1, 0.4]])
    expected = project_points_to_mesh_with_faces(points, mesh)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = _project_with_workers(points, mesh, executor, 2)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], [1, 0, 0])


def test_cli_forwards_diagnostic_and_reference_options(monkeypatch):
    import src.landmark_error_diagnostics as diagnostic
    from train_pipeline import build_parser

    captured = []
    monkeypatch.setattr(diagnostic, "main", lambda values: captured.extend(values))
    args = build_parser().parse_args([
        "analyze-landmark-errors", "--checkpoint-path", "model.pt", "--prior-path", "prior.npz",
        "--prior-manifest", "prior.json", "--folds-json", "folds.json", "--predictions-json", "centers.json",
        "--calibration-json", "crop.json", "--reference-report", "reference.json",
        "--frame-max-surface-distance-mm", "0.25", "--run-seed", "43", "--output", "errors.json",
    ])
    args.function(args)
    forwarded = diagnostic.build_parser().parse_args(captured)
    assert forwarded.reference_report == "reference.json"
    assert forwarded.frame_max_surface_distance_mm == 0.25
    assert forwarded.run_seed == 43
    assert forwarded.components == 32
    assert forwarded.projection_workers == 10
