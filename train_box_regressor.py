"""Train the Crop Box Regressor to find ear bounding boxes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.torch_dataset import PinnaEarBoxDataset, load_subject_ids, split_subject_ids
from src.pointnet2_model import PointNet2BoxRegressor
from src.dataset import Dataset as MeshLandmarkDataset
from src.ear_crop import fit_crop_config_from_training_landmarks
from train_pointnet2 import set_seed, make_device, make_optimizer, make_scheduler, save_checkpoint


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    amp: bool = False,
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_center_err = 0.0
    total_size_err = 0.0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=training and amp)

    for batch in loader:
        points = batch["points"].to(device)
        target_box = batch["box"].to(device)
        batch_size = points.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=training and amp):
                pred_box = model(points)
                
                # Split predictions into center (first 3) and size (last 3)
                pred_center, pred_size = pred_box[:, :3], pred_box[:, 3:]
                target_center, target_size = target_box[:, :3], target_box[:, 3:]
                
                # Calculate loss (Weight size loss slightly less to prioritize centering)
                loss_center = F.smooth_l1_loss(pred_center, target_center)
                loss_size = F.smooth_l1_loss(pred_size, target_size)
                loss = loss_center + (0.5 * loss_size)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Track absolute errors (normalized scale)
        with torch.no_grad():
            center_err = torch.linalg.norm(pred_center - target_center, dim=-1).mean()
            size_err = torch.linalg.norm(pred_size - target_size, dim=-1).mean()

        total_loss += loss.item() * batch_size
        total_center_err += center_err.item() * batch_size
        total_size_err += size_err.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "center_error": total_center_err / max(total_samples, 1),
        "size_error": total_size_err / max(total_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Ear Box Regressor")
    parser.add_argument("--mesh-dir", default="data/mesh")
    parser.add_argument("--landmarks-dir", default="data/landmarks")
    parser.add_argument("--checkpoint-dir", default="checkpoints_box")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ear-points", type=int, default=8192)
    parser.add_argument("--broad-margin", type=float, default=0.40) # Large margin for input
    parser.add_argument("--tight-margin", type=float, default=0.15) # Tight margin for target
    args, _ = parser.parse_known_args()

    device = make_device("auto")
    set_seed(0)

    # Load Data
    base_dataset = MeshLandmarkDataset(args.mesh_dir, args.landmarks_dir)
    train_ids, val_ids = split_subject_ids(base_dataset, 0.2, 0)

    # Fit the current crop to serve as the input space
    print("Fitting broad crops for input...")
    broad_crop_config = fit_crop_config_from_training_landmarks(
        base_dataset, train_ids, margin=args.broad_margin
    )

    # Create Box Datasets
    train_dataset = PinnaEarBoxDataset(
        args.mesh_dir, args.landmarks_dir, broad_crop_config, 
        ear_points=args.ear_points, subject_ids=train_ids, box_margin=args.tight_margin
    )
    val_dataset = PinnaEarBoxDataset(
        args.mesh_dir, args.landmarks_dir, broad_crop_config, 
        ear_points=args.ear_points, subject_ids=val_ids, box_margin=args.tight_margin
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize Model
    model = PointNet2BoxRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training Loop
    best_loss = float("inf")
    print(f"Starting Box Regressor Training on {device}...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer, amp=True)
        val_metrics = run_epoch(model, val_loader, device, amp=True)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Center Err: {val_metrics['center_error']:.4f} "
            f"Val Size Err: {val_metrics['size_error']:.4f}"
        )

        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save(model.state_dict(), checkpoint_dir / "best_box_model.pt")

if __name__ == "__main__":
    main()