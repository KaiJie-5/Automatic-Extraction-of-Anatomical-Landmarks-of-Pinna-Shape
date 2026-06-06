"""Train a PointNet++ regressor for pinna landmark extraction."""

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import Dataset as MeshLandmarkDataset
from src.pointnet2_model import PointNet2LandmarkRegressor, default_model_config
from src.torch_dataset import PinnaPointCloudDataset, load_subject_ids, split_subject_ids


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_nested_ints(value: str) -> List[List[int]]:
    return [parse_int_list(group) for group in value.split(";") if group.strip()]


def parse_nested_floats(value: str) -> List[List[float]]:
    return [parse_float_list(group) for group in value.split(";") if group.strip()]


def parse_msg_mlps(value: str) -> List[List[List[int]]]:
    layers = []
    for layer in value.split(";"):
        if not layer.strip():
            continue
        layers.append([parse_int_list(scale) for scale in layer.split("|") if scale.strip()])
    return layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_model_config(args: argparse.Namespace) -> dict:
    config = default_model_config()
    config.update(
        {
            "num_landmarks": args.num_landmarks,
            "input_channels": 6,
            "use_normals": not args.no_normals,
            "variant": args.variant,
            "head_channels": parse_int_list(args.head_channels),
            "dropout": args.dropout,
            "ssg_npoints": parse_int_list(args.ssg_npoints),
            "ssg_radii": parse_float_list(args.ssg_radii),
            "ssg_nsamples": parse_int_list(args.ssg_nsamples),
            "ssg_mlps": parse_nested_ints(args.ssg_mlps),
            "msg_npoints": parse_int_list(args.msg_npoints),
            "msg_radii": parse_nested_floats(args.msg_radii),
            "msg_nsamples": parse_nested_ints(args.msg_nsamples),
            "msg_mlps": parse_msg_mlps(args.msg_mlps),
            "msg_global_mlp": parse_int_list(args.msg_global_mlp),
        }
    )
    return config


def make_optimizer(args: argparse.Namespace, parameters: Iterable[torch.nn.Parameter]):
    name = args.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def make_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    name = args.scheduler.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.step_gamma
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def make_criterion(args: argparse.Namespace) -> Optional[torch.nn.Module]:
    name = args.loss.lower()
    if name == "mean_distance":
        return None
    if name == "smooth_l1":
        try:
            return torch.nn.SmoothL1Loss(beta=args.smooth_l1_beta)
        except TypeError:
            return torch.nn.SmoothL1Loss()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "l1":
        return torch.nn.L1Loss()
    raise ValueError(f"Unsupported loss: {args.loss}")


def mean_landmark_distance(
    pred_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    centroid: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scale = scale.view(-1, 1, 1)
    centroid = centroid.view(-1, 1, 3)
    pred = pred_normalized * scale + centroid
    target = target_normalized * scale + centroid
    return torch.linalg.norm(pred - target, dim=-1).mean()


def compute_training_loss(
    criterion: Optional[torch.nn.Module],
    loss_name: str,
    pred_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    centroid: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Compute the selected training loss.

    The "mean_distance" option matches the official metric in torch form by
    de-normalizing predictions and targets before computing Euclidean distance.
    """
    if loss_name == "mean_distance":
        return mean_landmark_distance(pred_normalized, target_normalized, centroid, scale)
    if criterion is None:
        raise ValueError(f"criterion cannot be None for loss '{loss_name}'")
    return criterion(pred_normalized, target_normalized)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Optional[torch.nn.Module],
    loss_name: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_md = 0.0
    total_samples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=training and amp)

    for batch in loader:
        points = batch["points"].to(device=device, dtype=torch.float32)
        target = batch["landmarks"].to(device=device, dtype=torch.float32)
        centroid = batch["centroid"].to(device=device, dtype=torch.float32)
        scale = batch["scale"].to(device=device, dtype=torch.float32)
        batch_size = points.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=training and amp):
                pred = model(points)
                loss = compute_training_loss(
                    criterion, loss_name, pred, target, centroid, scale
                )

            if training:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        md = mean_landmark_distance(pred.detach(), target, centroid, scale)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_md += float(md.detach().cpu()) * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "mean_distance": total_md / max(total_samples, 1),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_config: dict,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "num_points": args.num_points,
            "seed": args.seed,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--mesh-dir", default="data/mesh")
    parser.add_argument("--landmarks-dir", default="data/landmarks")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--train-split-file", default=None)
    parser.add_argument("--val-split-file", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--num-landmarks", type=int, default=170)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--variant", choices=["ssg", "msg"], default="ssg")
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument("--head-channels", default="512,256")
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--ssg-npoints", default="512,128")
    parser.add_argument("--ssg-radii", default="0.2,0.4")
    parser.add_argument("--ssg-nsamples", default="32,64")
    parser.add_argument("--ssg-mlps", default="64,64,128;128,128,256;256,512,1024")
    parser.add_argument("--msg-npoints", default="512,128")
    parser.add_argument("--msg-radii", default="0.1,0.2,0.4;0.2,0.4,0.8")
    parser.add_argument("--msg-nsamples", default="16,32,128;32,64,128")
    parser.add_argument(
        "--msg-mlps",
        default="32,32,64|64,64,128|64,96,128;64,64,128|128,128,256|128,128,256",
    )
    parser.add_argument("--msg-global-mlp", default="256,512,1024")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--loss",
        choices=["smooth_l1", "mse", "l1", "mean_distance"],
        default="smooth_l1",
    )
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default="cosine")
    parser.add_argument("--step-size", type=int, default=50)
    parser.add_argument("--step-gamma", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="auto")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = make_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    base_dataset = MeshLandmarkDataset(args.mesh_dir, args.landmarks_dir)
    train_ids = load_subject_ids(args.train_split_file)
    val_ids = load_subject_ids(args.val_split_file)
    if train_ids is None and val_ids is None:
        train_ids, val_ids = split_subject_ids(base_dataset, args.val_ratio, args.seed)
    elif train_ids is None or val_ids is None:
        raise ValueError("Provide both --train-split-file and --val-split-file, or neither")

    train_dataset = PinnaPointCloudDataset(
        args.mesh_dir,
        args.landmarks_dir,
        num_points=args.num_points,
        seed=args.seed,
        subject_ids=train_ids,
    )
    val_dataset = (
        PinnaPointCloudDataset(
            args.mesh_dir,
            args.landmarks_dir,
            num_points=args.num_points,
            seed=args.seed + 100000,
            subject_ids=val_ids,
        )
        if val_ids
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        if val_dataset is not None
        else None
    )

    model_config = make_model_config(args)
    model = PointNet2LandmarkRegressor(**model_config).to(device)
    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model.parameters())
    scheduler = make_scheduler(args, optimizer)

    checkpoint_dir = Path(args.checkpoint_dir)
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            args.loss,
            device,
            optimizer=optimizer,
            amp=amp_enabled,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = (
            run_epoch(model, val_loader, criterion, args.loss, device, amp=amp_enabled)
            if val_loader is not None
            else None
        )
        if scheduler is not None:
            scheduler.step()

        score = val_metrics["mean_distance"] if val_metrics is not None else train_metrics["loss"]
        latest_metrics = {"train": train_metrics, "val": val_metrics}
        save_checkpoint(
            checkpoint_dir / "last_model.pt", model, model_config, args, epoch, latest_metrics
        )
        if score < best_score:
            best_score = score
            save_checkpoint(
                checkpoint_dir / "best_model.pt",
                model,
                model_config,
                args,
                epoch,
                latest_metrics,
            )

        val_text = (
            f" val_loss={val_metrics['loss']:.6f} val_md={val_metrics['mean_distance']:.6f}"
            if val_metrics is not None
            else ""
        )
        print(
            f"epoch={epoch:04d}"
            f" train_loss={train_metrics['loss']:.6f}"
            f" train_md={train_metrics['mean_distance']:.6f}"
            f"{val_text}"
        )


if __name__ == "__main__":
    main()
