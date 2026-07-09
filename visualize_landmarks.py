"""Overlay ground-truth and predicted landmarks on the full mesh.

For every validation subject that has a predicted landmark file, this builds a
single self-contained PLY containing:
    - the full mesh                          (gray)
    - ground-truth landmarks (left + right)  (green spheres)
    - predicted landmarks (left + right)     (red spheres)

Open the resulting `<subject>_overlay.ply` in MeshLab / CloudCompare and you
see the mesh with both landmark sets colored, so you can judge placement at a
glance. A colored point-only PLY is also written for lighter viewing.

Ground-truth landmarks come from the CSVs in --landmarks-dir
(`<subject>_left_ear_landmarks.csv` / `<subject>_right_ear_landmarks.csv`),
predictions from --pred-dir (`<subject>_full_pred_landmarks.csv`, or the
per-ear `<subject>_<ear>_pred_landmarks.csv`), and meshes from --mesh-dir.
All coordinates are assumed to be in the same original full-mesh frame.
"""

import argparse
import csv
import colorsys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trimesh

EAR_NAMES = ("left", "right")

GT_COLOR = np.array([40, 200, 40, 255], dtype=np.uint8)      # green
PRED_COLOR = np.array([220, 40, 40, 255], dtype=np.uint8)    # red
MESH_COLOR = np.array([185, 185, 185, 255], dtype=np.uint8)  # gray


def load_landmark_csv(path: Path) -> Optional[np.ndarray]:
    """Load an (N, 3) landmark CSV in 'index, [x y z]' format."""
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        coords = [
            np.fromstring(coordinate_str.strip("[]"), sep=" ")
            for _, coordinate_str in reader
        ]
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64)


def load_ground_truth(landmarks_dir: Path, subject_id: str) -> Optional[np.ndarray]:
    parts = []
    for ear in EAR_NAMES:
        arr = load_landmark_csv(landmarks_dir / f"{subject_id}_{ear}_ear_landmarks.csv")
        if arr is None:
            return None
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def load_prediction(pred_dir: Path, subject_id: str) -> Optional[np.ndarray]:
    combined = load_landmark_csv(pred_dir / f"{subject_id}_full_pred_landmarks.csv")
    if combined is not None:
        return combined
    parts = []
    for ear in EAR_NAMES:
        arr = load_landmark_csv(pred_dir / f"{subject_id}_{ear}_pred_landmarks.csv")
        if arr is None:
            return None
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def find_subjects(pred_dir: Path) -> List[str]:
    subjects = set()
    for path in pred_dir.glob("*_full_pred_landmarks.csv"):
        subjects.add(path.name[: -len("_full_pred_landmarks.csv")])
    for path in pred_dir.glob("*_left_pred_landmarks.csv"):
        subjects.add(path.name[: -len("_left_pred_landmarks.csv")])
    return sorted(subjects)


def make_landmark_spheres(
    points: np.ndarray, radius: float, color: np.ndarray, subdivisions: int = 1
) -> trimesh.Trimesh:
    """Return one mesh: a colored sphere placed at each landmark point."""
    template = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    all_v = []
    all_f = []
    v_offset = 0
    for p in points:
        v = template.vertices + p
        all_v.append(v)
        all_f.append(template.faces + v_offset)
        v_offset += len(template.vertices)
    vertices = np.concatenate(all_v, axis=0)
    faces = np.concatenate(all_f, axis=0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = np.tile(color, (len(vertices), 1))
    return mesh


def make_point_cloud(
    gt: np.ndarray, pred: np.ndarray
) -> trimesh.points.PointCloud:
    pts = np.concatenate([gt, pred], axis=0)
    colors = np.concatenate(
        [
            np.tile(GT_COLOR, (len(gt), 1)),
            np.tile(PRED_COLOR, (len(pred), 1)),
        ],
        axis=0,
    )
    return trimesh.points.PointCloud(vertices=pts, colors=colors)


def auto_radius(mesh: trimesh.Trimesh, fraction: float) -> float:
    extents = mesh.bounding_box.extents
    diag = float(np.linalg.norm(extents))
    return max(diag * fraction, 1e-6)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", default="data/mesh")
    parser.add_argument("--landmarks-dir", default="data/landmarks")
    parser.add_argument("--pred-dir", default="predictions_meshnet/val")
    parser.add_argument("--output-dir", default="landmark_overlays/val")
    parser.add_argument(
        "--point-radius",
        type=float,
        default=0.0,
        help="Sphere radius in mesh units. 0 = auto (fraction of mesh size).",
    )
    parser.add_argument(
        "--radius-fraction",
        type=float,
        default=0.004,
        help="Auto radius as this fraction of the mesh bounding-box diagonal.",
    )
    parser.add_argument(
        "--sphere-subdivisions",
        type=int,
        default=1,
        help="Icosphere subdivisions per landmark (higher = smoother, heavier).",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Only export the colored landmark point cloud, not the merged mesh.",
    )
    parser.add_argument(
        "--subject-ids",
        default=None,
        help="Comma-separated subject ids to restrict to (default: all in pred-dir).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    mesh_dir = Path(args.mesh_dir)
    landmarks_dir = Path(args.landmarks_dir)
    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.subject_ids:
        subjects = [s.strip() for s in args.subject_ids.split(",") if s.strip()]
    else:
        subjects = find_subjects(pred_dir)

    if not subjects:
        print(f"No predicted landmark files found in {pred_dir}")
        return

    print(f"Overlaying {len(subjects)} subject(s): GT=green, predicted=red")
    written = 0
    for subject_id in subjects:
        mesh_path = mesh_dir / f"{subject_id}.ply"
        if not mesh_path.exists():
            print(f"  {subject_id}: SKIP (mesh not found: {mesh_path.name})")
            continue

        gt = load_ground_truth(landmarks_dir, subject_id)
        pred = load_prediction(pred_dir, subject_id)
        if pred is None:
            print(f"  {subject_id}: SKIP (no predicted landmarks)")
            continue

        mesh = trimesh.load(mesh_path, force="mesh")
        radius = args.point_radius if args.point_radius > 0 else auto_radius(
            mesh, args.radius_fraction
        )

        # Colored point-only PLY (light).
        if gt is not None:
            pc = make_point_cloud(gt, pred)
        else:
            pc = trimesh.points.PointCloud(
                vertices=pred, colors=np.tile(PRED_COLOR, (len(pred), 1))
            )
        pc.export(output_dir / f"{subject_id}_landmarks_points.ply")

        # Merged mesh + spheres PLY (single self-contained file).
        if not args.no_mesh:
            mesh_gray = mesh.copy()
            mesh_gray.visual.vertex_colors = np.tile(MESH_COLOR, (len(mesh_gray.vertices), 1))
            parts = [mesh_gray]
            if gt is not None:
                parts.append(make_landmark_spheres(gt, radius, GT_COLOR, args.sphere_subdivisions))
            parts.append(make_landmark_spheres(pred, radius, PRED_COLOR, args.sphere_subdivisions))
            scene_mesh = trimesh.util.concatenate(parts)
            scene_mesh.export(output_dir / f"{subject_id}_overlay.ply")

        gt_n = 0 if gt is None else len(gt)
        print(
            f"  {subject_id}: mesh + {gt_n} GT (green) + {len(pred)} pred (red), "
            f"radius={radius:.3f}"
        )
        written += 1

    print(f"\nWrote overlays for {written} subject(s) to: {output_dir}")
    print("Open <subject>_overlay.ply in MeshLab/CloudCompare (mesh gray, GT green, pred red).")


if __name__ == "__main__":
    main()