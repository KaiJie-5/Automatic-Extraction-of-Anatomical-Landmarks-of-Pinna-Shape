"""Visualize point importance for a trained PointNet++ landmark regressor."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

from src.dataset import Dataset as MeshLandmarkDataset
from src.pointnet2_model import PointNet2LandmarkRegressor, default_model_config
from src.preprocessing import (
    compute_mesh_normalization,
    make_landmark_target,
    normalize_point_features,
    sample_mesh_surface,
)


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(
    checkpoint_path: str, device: torch.device
) -> Tuple[PointNet2LandmarkRegressor, dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = default_model_config()
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        model_config.update(checkpoint["model_config"])

    model = PointNet2LandmarkRegressor(**model_config).to(device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_config, checkpoint if isinstance(checkpoint, dict) else {}


def load_mesh_and_target(args: argparse.Namespace):
    if args.mesh_path is not None:
        mesh = trimesh.load(args.mesh_path)
        return mesh, None, Path(args.mesh_path).stem

    if args.subject_id is None:
        raise ValueError("Provide either --mesh-path or --subject-id")

    dataset = MeshLandmarkDataset(args.mesh_dir, args.landmarks_dir)
    id_to_index = {dataset.get_identifier(idx): idx for idx in range(len(dataset))}
    if args.subject_id not in id_to_index:
        raise ValueError(f"Unknown subject id: {args.subject_id}")

    mesh, left, right = dataset[id_to_index[args.subject_id]]
    return mesh, (left, right), args.subject_id


def prepare_points(mesh: trimesh.Trimesh, num_points: int, seed: int):
    transform = compute_mesh_normalization(mesh)
    point_features = sample_mesh_surface(mesh, num_points=num_points, seed=seed)
    points_xyz = point_features[:, :3].copy()
    normalized_features = normalize_point_features(point_features, transform)
    return transform, points_xyz, normalized_features


def official_mean_distance_torch(
    pred_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    pred = pred_normalized * scale
    target = target_normalized * scale
    return torch.linalg.norm(pred - target, dim=-1).mean()


def gradient_importance(
    model: PointNet2LandmarkRegressor,
    points: torch.Tensor,
    target: Optional[torch.Tensor],
    scale: float,
    objective: str,
) -> Tuple[np.ndarray, torch.Tensor]:
    points = points.clone().detach().requires_grad_(True)
    pred = model(points)

    if target is not None:
        score = official_mean_distance_torch(pred, target, scale)
    elif objective == "l2":
        score = torch.linalg.norm(pred, dim=-1).mean()
    else:
        score = pred.abs().mean()

    model.zero_grad(set_to_none=True)
    score.backward()
    grads = points.grad.detach()[0, :, :3]
    importance = torch.linalg.norm(grads, dim=-1).cpu().numpy()
    return importance, pred.detach()


def occlusion_importance(
    model: PointNet2LandmarkRegressor,
    points: torch.Tensor,
    batch_size: int,
    occlusion_group_size: int,
    occlusion_value: str,
) -> Tuple[np.ndarray, torch.Tensor]:
    with torch.no_grad():
        baseline = model(points)

    num_points = points.shape[1]
    group_size = max(1, int(occlusion_group_size))
    group_starts = list(range(0, num_points, group_size))
    group_scores = []

    if occlusion_value == "mean":
        fill = torch.zeros_like(points[:, :1])
        fill[:, :, :3] = points[:, :, :3].mean(dim=1, keepdim=True)
    else:
        fill = torch.zeros_like(points[:, :1])

    for start in range(0, len(group_starts), batch_size):
        batch_starts = group_starts[start : start + batch_size]
        occluded = points.repeat(len(batch_starts), 1, 1)
        for row, point_start in enumerate(batch_starts):
            point_end = min(point_start + group_size, num_points)
            occluded[row, point_start:point_end, :] = fill[0]

        with torch.no_grad():
            pred = model(occluded)
            diff = torch.linalg.norm(pred - baseline.repeat(len(batch_starts), 1, 1), dim=-1)
            group_scores.extend(diff.mean(dim=1).cpu().numpy().tolist())

    importance = np.zeros(num_points, dtype=np.float32)
    for score, point_start in zip(group_scores, group_starts):
        point_end = min(point_start + group_size, num_points)
        importance[point_start:point_end] = score
    return importance, baseline.detach()


def normalize_importance(importance: np.ndarray) -> np.ndarray:
    importance = np.asarray(importance, dtype=np.float32)
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(importance.min())
    max_value = float(importance.max())
    if max_value <= min_value:
        return np.zeros_like(importance, dtype=np.float32)
    return ((importance - min_value) / (max_value - min_value)).astype(np.float32)


def colorize_importance(importance: np.ndarray) -> np.ndarray:
    """Blue to cyan to yellow to red heatmap without external plotting deps."""
    values = normalize_importance(importance)
    colors = np.zeros((len(values), 4), dtype=np.uint8)
    colors[:, 3] = 255

    low = values < 0.33
    mid = (values >= 0.33) & (values < 0.66)
    high = values >= 0.66

    colors[low, 1] = np.clip(values[low] / 0.33 * 255, 0, 255).astype(np.uint8)
    colors[low, 2] = 255

    colors[mid, 0] = np.clip((values[mid] - 0.33) / 0.33 * 255, 0, 255).astype(np.uint8)
    colors[mid, 1] = 255
    colors[mid, 2] = np.clip((1.0 - (values[mid] - 0.33) / 0.33) * 255, 0, 255).astype(
        np.uint8
    )

    colors[high, 0] = 255
    colors[high, 1] = np.clip((1.0 - (values[high] - 0.66) / 0.34) * 255, 0, 255).astype(
        np.uint8
    )
    return colors


def export_point_cloud(path: Path, xyz: np.ndarray, importance: np.ndarray) -> None:
    colors = colorize_importance(importance)
    point_cloud = trimesh.points.PointCloud(vertices=xyz, colors=colors)
    point_cloud.export(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--mesh-path", default=None)
    parser.add_argument("--subject-id", default=None)
    parser.add_argument(
        "--mesh-dir",
        default="/iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape/data/mesh",
    )
    parser.add_argument(
        "--landmarks-dir",
        default="/iridisfs/home/kjl1a21/Automatic-Extraction-of-Anatomical-Landmarks-of-Pinna-Shape/data/landmarks",
    )
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["gradient", "occlusion"], default="occlusion")
    parser.add_argument("--gradient-objective", choices=["l1", "l2"], default="l2")
    parser.add_argument("--occlusion-group-size", type=int, default=128)
    parser.add_argument("--occlusion-batch-size", type=int, default=16)
    parser.add_argument("--occlusion-value", choices=["zero", "mean"], default="mean")
    parser.add_argument("--output-dir", default="importance_outputs")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    device = make_device(args.device)
    model, model_config, checkpoint = load_model(args.checkpoint_path, device)
    mesh, landmarks, identifier = load_mesh_and_target(args)

    checkpoint_num_points = checkpoint.get("num_points") if isinstance(checkpoint, dict) else None
    num_points = args.num_points or checkpoint_num_points or 16384
    transform, original_xyz, normalized_features = prepare_points(mesh, int(num_points), args.seed)
    points = torch.from_numpy(normalized_features).unsqueeze(0).to(device=device, dtype=torch.float32)

    target_tensor = None
    if landmarks is not None:
        target = make_landmark_target(landmarks[0], landmarks[1], transform)
        target_tensor = torch.from_numpy(target).unsqueeze(0).to(device=device, dtype=torch.float32)

    if args.method == "gradient":
        importance, pred = gradient_importance(
            model=model,
            points=points,
            target=target_tensor,
            scale=transform.scale,
            objective=args.gradient_objective,
        )
    else:
        importance, pred = occlusion_importance(
            model=model,
            points=points,
            batch_size=args.occlusion_batch_size,
            occlusion_group_size=args.occlusion_group_size,
            occlusion_value=args.occlusion_value,
        )

    importance = normalize_importance(importance)
    pred_denormalized = transform.denormalize_xyz(pred.squeeze(0).cpu().numpy()).astype(np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{identifier}_{args.method}_importance"
    ply_path = output_dir / f"{stem}.ply"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}_config.json"

    export_point_cloud(ply_path, original_xyz, importance)
    np.savez_compressed(
        npz_path,
        points_xyz=original_xyz,
        normalized_features=normalized_features,
        importance=importance,
        predicted_landmarks=pred_denormalized,
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint_path": args.checkpoint_path,
                "identifier": identifier,
                "method": args.method,
                "num_points": int(num_points),
                "seed": args.seed,
                "model_config": model_config,
                "output_ply": str(ply_path),
                "output_npz": str(npz_path),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(f"Wrote colored point cloud: {ply_path}")
    print(f"Wrote raw arrays: {npz_path}")
    print(f"Wrote config: {json_path}")


if __name__ == "__main__":
    main()
