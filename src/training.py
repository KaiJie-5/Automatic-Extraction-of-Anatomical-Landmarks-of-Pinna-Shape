"""Shared training loops for proposal v2 experiments."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import proposal_landmark_loss
from .precision import (
    autocast_context,
    grad_scaler_enabled,
    resolved_amp_dtype_name,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _autocast(
    device: torch.device,
    enabled: bool,
    amp_dtype: str = "auto",
):
    return autocast_context(device, enabled, amp_dtype)


def _json_dump(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _loader(dataset, batch_size: int, workers: int, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def probe_batch_size(
    model: torch.nn.Module,
    sample: Mapping[str, object],
    device: torch.device,
    candidates: Sequence[int] = (32, 16, 8, 4, 2, 1),
    amp: bool = True,
    amp_dtype: str = "auto",
) -> int:
    """Probe model forward/backward memory without changing learned parameters."""
    if device.type != "cuda":
        return 1
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for candidate in candidates:
        try:
            torch.cuda.empty_cache()
            if "face_features" in sample:
                face_features = sample["face_features"].unsqueeze(0).expand(candidate, -1, -1).contiguous().to(device)
                neighbors = sample["neighbors"].unsqueeze(0).expand(candidate, -1, -1).contiguous().to(device)
            else:
                points = sample["points"].unsqueeze(0).expand(candidate, -1, -1).contiguous().to(device)
            model.zero_grad(set_to_none=True)
            with _autocast(device, amp, amp_dtype):
                output = model(face_features, neighbors) if "face_features" in sample else model(points)
                output.float().square().mean().backward()
            model.zero_grad(set_to_none=True)
            model.load_state_dict(original)
            torch.cuda.empty_cache()
            return int(candidate)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
                model.load_state_dict(original)
                raise
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    model.load_state_dict(original)
    raise RuntimeError("model did not fit on the GPU even with physical batch size 1")


def _save_component_checkpoint(
    path: Path,
    component: str,
    model: torch.nn.Module,
    model_config: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: Mapping[str, object],
    data_config: Mapping[str, object],
    training_config: Mapping[str, object] | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "component_schema_version": 1,
            "component": component,
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_config),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "data_config": dict(data_config),
            "training_config": dict(training_config or {}),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
        },
        path,
    )


def train_locator(
    model: torch.nn.Module,
    train_dataset,
    validation_dataset,
    output_dir: str,
    model_config: Mapping[str, object],
    data_config: Mapping[str, object],
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 0,
    effective_batch_size: int = 32,
    workers: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    amp: bool = True,
    resume: bool = True,
    amp_dtype: str = "auto",
) -> Mapping[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    resolved_dtype = resolved_amp_dtype_name(device, amp, amp_dtype)
    runtime_config = {
        "amp": bool(amp),
        "amp_dtype": resolved_dtype,
        "grad_scaler_enabled": grad_scaler_enabled(device, amp, amp_dtype),
    }
    if batch_size <= 0:
        batch_size = probe_batch_size(
            model, train_dataset[0], device, amp=amp, amp_dtype=amp_dtype
        )
    accumulation = max(1, int(np.ceil(effective_batch_size / batch_size)))
    train_loader = _loader(train_dataset, batch_size, workers, True)
    validation_loader = _loader(validation_dataset, batch_size, workers, False) if validation_dataset else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler(
        enabled=runtime_config["grad_scaler_enabled"]
    )
    start_epoch = 1
    best = float("inf")
    stale = 0
    last_path = output / "last_locator.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        saved_runtime = checkpoint.get("training_config", {})
        if saved_runtime and saved_runtime.get("amp_dtype") != resolved_dtype:
            raise ValueError(
                "cannot resume locator with a different AMP dtype: "
                f"checkpoint={saved_runtime.get('amp_dtype')}, current={resolved_dtype}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["metrics"].get("best_center_error_mm", best))
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    history = []
    metrics_path = output / "metrics.json"
    if resume and metrics_path.exists():
        history = list(json.loads(metrics_path.read_text(encoding="utf-8")).get("history", []))
    if start_epoch > epochs:
        best_item = min(history, key=lambda item: item["validation_center_error_mm"])
        return {
            "best_center_error_mm": float(best_item["validation_center_error_mm"]),
            "best_epoch": int(best_item["epoch"]),
            "physical_batch_size": int(best_item["physical_batch_size"]),
            "gradient_accumulation": int(best_item["gradient_accumulation"]),
            **runtime_config,
        }
    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_count = 0
        for step, batch in enumerate(train_loader, 1):
            points = batch["points"].to(device)
            target = batch["center_correction"].to(device)
            with _autocast(device, amp, amp_dtype):
                prediction = model(points)
                loss = torch.linalg.norm(prediction.float() - target.float(), dim=-1).mean()
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_sum += float(loss.detach()) * len(points)
            train_count += len(points)

        validation_error = train_sum / max(train_count, 1)
        if validation_loader:
            validation_dataset.set_epoch(0)
            model.eval()
            error_sum = 0.0
            count = 0
            with torch.no_grad():
                for batch in validation_loader:
                    points = batch["points"].to(device)
                    target = batch["center_correction"].to(device)
                    prediction = model(points)
                    errors = torch.linalg.norm(prediction.float() - target.float(), dim=-1)
                    error_sum += float(errors.sum())
                    count += len(points)
            validation_error = error_sum / max(count, 1)
        scheduler.step()
        improved = validation_error < best
        if improved:
            best = validation_error
            stale = 0
        else:
            stale += 1
        metrics = {
            "train_center_error_mm": train_sum / max(train_count, 1),
            "validation_center_error_mm": validation_error,
            "best_center_error_mm": best,
            "physical_batch_size": batch_size,
            "gradient_accumulation": accumulation,
            **runtime_config,
        }
        history.append({"epoch": epoch, **metrics})
        _save_component_checkpoint(last_path, "locator", model, model_config, optimizer, scheduler, epoch, metrics, data_config, runtime_config, scaler)
        if improved:
            _save_component_checkpoint(output / "best_locator.pt", "locator", model, model_config, optimizer, scheduler, epoch, metrics, data_config, runtime_config, scaler)
        _json_dump(metrics_path, {"history": history, "best": best})
        if validation_loader and stale >= patience:
            break
    return {"best_center_error_mm": best, "best_epoch": min(history, key=lambda item: item["validation_center_error_mm"])["epoch"], "physical_batch_size": batch_size, "gradient_accumulation": accumulation, **runtime_config}


def predict_locator(
    model: torch.nn.Module,
    dataset,
    broad_config: Mapping[str, object],
    device: torch.device,
    workers: int = 10,
) -> Mapping[str, list]:
    loader = _loader(dataset, 1, workers, False)
    initial = np.asarray(broad_config["initial_center"], dtype=np.float32)
    predictions = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            correction = model(batch["points"].to(device)).squeeze(0).float().cpu().numpy()
            key = f"{batch['identifier'][0]}:{batch['ear'][0]}"
            predictions[key] = (initial + correction).astype(np.float32).tolist()
    return predictions


def train_landmarks(
    model: torch.nn.Module,
    train_dataset,
    validation_dataset,
    output_dir: str,
    model_config: Mapping[str, object],
    data_config: Mapping[str, object],
    loss_weights: Mapping[str, float],
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 0,
    effective_batch_size: int = 32,
    workers: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    amp: bool = True,
    resume: bool = True,
    amp_dtype: str = "auto",
) -> Mapping[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    resolved_dtype = resolved_amp_dtype_name(device, amp, amp_dtype)
    runtime_config = {
        "amp": bool(amp),
        "amp_dtype": resolved_dtype,
        "grad_scaler_enabled": grad_scaler_enabled(device, amp, amp_dtype),
    }
    if batch_size <= 0:
        batch_size = probe_batch_size(
            model, train_dataset[0], device, amp=amp, amp_dtype=amp_dtype
        )
    accumulation = max(1, int(np.ceil(effective_batch_size / batch_size)))
    train_loader = _loader(train_dataset, batch_size, workers, True)
    validation_loader = _loader(validation_dataset, batch_size, workers, False) if validation_dataset else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler(
        enabled=runtime_config["grad_scaler_enabled"]
    )
    start_epoch = 1
    best = float("inf")
    stale = 0
    last_path = output / "last_landmarks.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        saved_runtime = checkpoint.get("training_config", {})
        if saved_runtime and saved_runtime.get("amp_dtype") != resolved_dtype:
            raise ValueError(
                "cannot resume landmarks with a different AMP dtype: "
                f"checkpoint={saved_runtime.get('amp_dtype')}, current={resolved_dtype}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["metrics"].get("best_md_mm", best))
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
    history = []
    metrics_path = output / "metrics.json"
    if resume and metrics_path.exists():
        history = list(json.loads(metrics_path.read_text(encoding="utf-8")).get("history", []))
    if start_epoch > epochs:
        best_item = min(history, key=lambda item: item["validation"]["mean_distance"])
        return {
            "best_md_mm": float(best_item["validation"]["mean_distance"]),
            "best_epoch": int(best_item["epoch"]),
            "physical_batch_size": int(best_item["physical_batch_size"]),
            "gradient_accumulation": int(best_item["gradient_accumulation"]),
            **runtime_config,
        }

    def run(loader, training: bool):
        model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
        sums = {"total": 0.0, "mean_distance": 0.0, "anchor": 0.0, "spacing": 0.0, "surface": 0.0}
        count = 0
        for step, batch in enumerate(loader, 1):
            if "face_features" in batch:
                face_features = batch["face_features"].to(device)
                neighbors = batch["neighbors"].to(device)
                batch_count = len(face_features)
            else:
                points = batch["points"].to(device)
                batch_count = len(points)
            target = batch["landmarks"].to(device)
            dense = batch.get("dense_surface")
            dense = dense.to(device) if dense is not None else None
            with torch.set_grad_enabled(training):
                with _autocast(device, amp, amp_dtype):
                    prediction = model(face_features, neighbors) if "face_features" in batch else model(points)
                    losses = proposal_landmark_loss(
                        prediction.float(), target.float(), batch["scale"].to(device), dense_surface=dense,
                        anchor_weight=float(loss_weights.get("anchor", 0.0)),
                        spacing_weight=float(loss_weights.get("spacing", 0.0)),
                        surface_weight=float(loss_weights.get("surface", 0.0)),
                    )
                if training:
                    scaler.scale(losses["total"] / accumulation).backward()
                    if step % accumulation == 0 or step == len(loader):
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
            for key in sums:
                sums[key] += float(losses[key].detach()) * batch_count
            count += batch_count
        return {key: value / max(count, 1) for key, value in sums.items()}

    for epoch in range(start_epoch, epochs + 1):
        train_dataset.set_epoch(epoch)
        train_metrics = run(train_loader, True)
        validation_metrics = train_metrics
        if validation_loader:
            validation_dataset.set_epoch(0)
            with torch.no_grad():
                validation_metrics = run(validation_loader, False)
        scheduler.step()
        score = validation_metrics["mean_distance"]
        improved = score < best
        if improved:
            best = score
            stale = 0
        else:
            stale += 1
        metrics = {
            "train": train_metrics,
            "validation": validation_metrics,
            "best_md_mm": best,
            "physical_batch_size": batch_size,
            "gradient_accumulation": accumulation,
            **runtime_config,
        }
        history.append({"epoch": epoch, **metrics})
        _save_component_checkpoint(last_path, "landmarks", model, model_config, optimizer, scheduler, epoch, metrics, data_config, runtime_config, scaler)
        if improved:
            _save_component_checkpoint(output / "best_landmarks.pt", "landmarks", model, model_config, optimizer, scheduler, epoch, metrics, data_config, runtime_config, scaler)
        _json_dump(metrics_path, {"history": history, "best": best})
        if validation_loader and stale >= patience:
            break
    best_item = min(history, key=lambda item: item["validation"]["mean_distance"])
    return {"best_md_mm": best, "best_epoch": best_item["epoch"], "physical_batch_size": batch_size, "gradient_accumulation": accumulation, **runtime_config}


def benchmark_inference(
    function: Callable[[], object], warmup: int = 10, repeats: int = 100
) -> Mapping[str, float]:
    for _ in range(warmup):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    return {"median_ms": float(np.median(timings)), "mean_ms": float(np.mean(timings))}
