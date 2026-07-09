import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from src.dataset import Dataset as MeshLandmarkDataset
from src.meshnet_model import MeshNetLandmarkRegressor
from src.meshnet_dataset import PinnaPrecropMeshDataset, EAR_NAMES


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_split_ids(checkpoint_path: Path, split: str) -> Optional[List[str]]:
    """Read train/val subject ids from run_config.json next to the checkpoint."""
    run_config_path = checkpoint_path.parent / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"run_config.json not found next to checkpoint ({run_config_path}). "
            "Pass --subject-ids explicitly, or use --split all."
        )
    with run_config_path.open("r", encoding="utf-8") as handle:
        run_config = json.load(handle)
    data_split = run_config.get("data_split", {})
    train_ids = data_split.get("train_ids", [])
    val_ids = data_split.get("val_ids", [])
    if split == "val":
        return list(val_ids)
    if split == "train":
        return list(train_ids)
    if split == "all":
        return None
    raise ValueError(f"Unknown split: {split}")


def write_landmark_csv(path: Path, coords: np.ndarray) -> None:
    """Write (N, 3) coordinates in the ground-truth CSV format: index, '[x y z]'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for i, (x, y, z) in enumerate(coords):
            writer.writerow([i, f"[{x:.6f} {y:.6f} {z:.6f}]"])


def load_ground_truth(landmarks_dir: Path, subject_id: str, ear: str) -> Optional[np.ndarray]:
    path = landmarks_dir / f"{subject_id}_{ear}_ear_landmarks.csv"
    if not path.exists():
        return None
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        coords = [
            np.fromstring(coordinate_str.strip("[]"), sep=" ")
            for _, coordinate_str in reader
        ]
    return np.asarray(coords, dtype=np.float32)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--mesh-dir", default="data/mesh")
    parser.add_argument("--landmarks-dir", default="data/landmarks")
    parser.add_argument(
        "--cropped-dir",
        default=None,
        help="Cropped ear meshes dir. Defaults to the value stored in the checkpoint.",
    )
    parser.add_argument("--output-dir", default="predictions_meshnet")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument(
        "--subject-ids",
        default=None,
        help="Comma-separated subject ids to override the split selection.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Override max_faces. Defaults to the value stored in the checkpoint.",
    )
    parser.add_argument(
        "--face-cache-dir",
        default=None,
        help=(
            "Reuse the face cache built during training (e.g. cache/meshnet_faces) "
            "so the exact same sampled faces are fed to the model. Strongly "
            "recommended for reproducing the training-time validation inputs."
        ),
    )
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    device = make_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint["model_config"]
    input_mode = checkpoint.get("input_mode", "precropped")
    if input_mode != "precropped":
        raise ValueError(f"This script only supports precropped MeshNet checkpoints, got {input_mode}")

    cropped_dir = args.cropped_dir or checkpoint.get("cropped_dir")
    if not cropped_dir:
        raise ValueError("cropped-dir not in checkpoint; pass --cropped-dir explicitly.")
    max_faces = args.max_faces or checkpoint.get("max_faces", 1024)
    mirror_right_ear = bool(checkpoint.get("mirror_right_ear", False))
    seed = int(checkpoint.get("seed", 0))
    best_epoch = checkpoint.get("epoch")

    if args.subject_ids:
        subject_ids: Optional[List[str]] = [
            s.strip() for s in args.subject_ids.split(",") if s.strip()
        ]
    else:
        subject_ids = load_split_ids(checkpoint_path, args.split)

    # Reproduce the EXACT training-time dataset construction:
    #   - train split used seed = args.seed
    #   - val split   used seed = args.seed + 100000  (see train_meshnet.py)
    #   - augment always off for evaluation
    #   - same mirror_right_ear as training (read from checkpoint)
    #   - same face cache -> identical sampled faces as training
    dataset_seed = seed + 100000 if args.split == "val" else seed
    dataset = PinnaPrecropMeshDataset(
        args.mesh_dir,
        args.landmarks_dir,
        cropped_dir=cropped_dir,
        max_faces=max_faces,
        seed=dataset_seed,
        subject_ids=subject_ids,
        mirror_right_ear=mirror_right_ear,
        augment=False,
        face_cache_dir=args.face_cache_dir,
    )

    model = MeshNetLandmarkRegressor(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    landmarks_dir = Path(args.landmarks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-subject predictions so we can also write a combined 170-pt file.
    per_subject: dict = {}
    metric_rows: List[dict] = []
    all_distances: List[np.ndarray] = []

    print(f"Loaded checkpoint: {checkpoint_path}")
    if best_epoch is not None:
        print(f"Checkpoint epoch: {best_epoch}")
    split_source = (
        "run_config.json (exact training split)"
        if not args.subject_ids
        else "--subject-ids override"
    )
    print(f"Split: {args.split} | source: {split_source}")
    print(f"Base seed (from checkpoint): {seed} | dataset seed used: {dataset_seed}")
    print(f"max_faces: {max_faces} | mirror_right_ear: {mirror_right_ear} | "
          f"face_cache: {args.face_cache_dir or 'none (recomputed from seed)'}")
    print(f"Subjects: {'all' if subject_ids is None else len(subject_ids)} | "
          f"samples (ears): {len(dataset)}")

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            subject_id = sample["identifier"]
            ear = sample["ear"]
            centroid = sample["centroid"].numpy().astype(np.float32)
            scale = float(sample["scale"])
            mirrored = bool(sample["mirrored"])

            centers = sample["centers"].unsqueeze(0).to(device=device, dtype=torch.float32)
            corners = sample["corners"].unsqueeze(0).to(device=device, dtype=torch.float32)
            normals = sample["normals"].unsqueeze(0).to(device=device, dtype=torch.float32)
            neighbor_index = sample["neighbor_index"].unsqueeze(0).to(device=device)

            pred = model(centers, corners, normals, neighbor_index)  # (1, 85, 3)
            pred = pred.squeeze(0).cpu().numpy().astype(np.float32)

            # Undo the right-ear Y mirror applied by the dataset, then
            # denormalize from the full-mesh normalized frame to original coords.
            if mirrored:
                pred[:, 1] *= -1.0
            pred_original = pred * scale + centroid

            write_landmark_csv(
                output_dir / f"{subject_id}_{ear}_pred_landmarks.csv", pred_original
            )
            per_subject.setdefault(subject_id, {})[ear] = pred_original

            # Error vs ground truth (in original mm coordinates).
            gt = load_ground_truth(landmarks_dir, subject_id, ear)
            row = {"subject": subject_id, "ear": ear, "num_landmarks": len(pred_original)}
            if gt is not None and gt.shape == pred_original.shape:
                dist = np.linalg.norm(pred_original - gt, axis=1)
                all_distances.append(dist)
                row.update(
                    {
                        "mean_distance_mm": float(dist.mean()),
                        "max_distance_mm": float(dist.max()),
                        "median_distance_mm": float(np.median(dist)),
                    }
                )
            else:
                row.update(
                    {"mean_distance_mm": "", "max_distance_mm": "", "median_distance_mm": ""}
                )
            metric_rows.append(row)
            md_text = (
                f" mean_md={row['mean_distance_mm']:.4f}mm"
                if isinstance(row["mean_distance_mm"], float)
                else ""
            )
            print(f"  {subject_id} {ear}: wrote {len(pred_original)} landmarks{md_text}")

    # Combined left+right (170) file per subject for full-mesh overlay.
    # Ordering matches the full-mode target convention: rows 0-84 = left ear,
    # rows 85-169 = right ear, all in original full-mesh coordinates.
    print("\nWriting combined 170-landmark files (left rows 0-84, right rows 85-169):")
    n_full = 0
    for subject_id in sorted(per_subject):
        ears = per_subject[subject_id]
        if all(ear in ears for ear in EAR_NAMES):
            combined = np.concatenate([ears["left"], ears["right"]], axis=0)
            out_path = output_dir / f"{subject_id}_full_pred_landmarks.csv"
            write_landmark_csv(out_path, combined)
            n_full += 1
            print(f"  {subject_id}: {combined.shape[0]} landmarks -> {out_path.name}")
        else:
            have = ", ".join(sorted(ears))
            print(f"  {subject_id}: SKIPPED combined file (only have: {have})")
    print(f"Wrote {n_full} combined 170-landmark file(s).")

    # Metrics CSV.
    metrics_path = output_dir / "prediction_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "ear",
                "num_landmarks",
                "mean_distance_mm",
                "median_distance_mm",
                "max_distance_mm",
            ],
        )
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": best_epoch,
        "split": args.split,
        "num_ears": len(metric_rows),
        "cropped_dir": str(cropped_dir),
        "max_faces": max_faces,
        "mirror_right_ear": mirror_right_ear,
    }
    if all_distances:
        stacked = np.concatenate(all_distances)
        summary["overall_mean_distance_mm"] = float(stacked.mean())
        summary["overall_median_distance_mm"] = float(np.median(stacked))
        summary["overall_max_distance_mm"] = float(stacked.max())
    with (output_dir / "prediction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nPredictions written to: {output_dir}")
    print(f"Per-ear metrics: {metrics_path}")
    if all_distances:
        print(
            f"Overall mean landmark distance: "
            f"{summary['overall_mean_distance_mm']:.4f} mm "
            f"(median {summary['overall_median_distance_mm']:.4f}, "
            f"max {summary['overall_max_distance_mm']:.4f})"
        )


if __name__ == "__main__":
    main()