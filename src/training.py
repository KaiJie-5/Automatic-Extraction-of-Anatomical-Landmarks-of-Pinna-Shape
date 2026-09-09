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


def _flatten_landmark_batch(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    dense_surface: torch.Tensor | None = None,
    heatmap_logits: torch.Tensor | None = None,
    heatmap_points: torch.Tensor | None = None,
    geodesic_distances: torch.Tensor | None = None,
    vote_offsets: torch.Tensor | None = None,
    curve_logits: torch.Tensor | None = None,
    curve_arc_coordinates: torch.Tensor | None = None,
    curve_landmark_fractions: torch.Tensor | None = None,
):
    """Flatten the optional paired-ear axis for unchanged per-ear losses."""
    if prediction.ndim == 3:
        return (
            prediction,
            target,
            scale,
            dense_surface,
            heatmap_logits,
            heatmap_points,
            geodesic_distances,
            vote_offsets,
            curve_logits,
            curve_arc_coordinates,
            curve_landmark_fractions,
            int(prediction.shape[0]),
        )
    if prediction.ndim != 4 or tuple(prediction.shape[1:3]) != (2, 85):
        raise ValueError(
            "landmark prediction must have shape (B, 85, 3) or "
            "(B, 2, 85, 3)"
        )
    if target.shape != prediction.shape:
        raise ValueError("paired landmark target shape does not match prediction")
    batch, ears = prediction.shape[:2]
    flattened_prediction = prediction.reshape(batch * ears, 85, 3)
    flattened_target = target.reshape(batch * ears, 85, 3)
    flattened_scale = scale.reshape(-1)
    if flattened_scale.numel() == batch:
        flattened_scale = flattened_scale.repeat_interleave(ears)
    if flattened_scale.numel() != batch * ears:
        raise ValueError("paired scale must contain one value per ear")
    flattened_dense = (
        dense_surface.reshape(batch * ears, *dense_surface.shape[2:])
        if dense_surface is not None
        else None
    )
    flattened_logits = (
        heatmap_logits.reshape(batch * ears, *heatmap_logits.shape[2:])
        if heatmap_logits is not None
        else None
    )
    flattened_points = (
        heatmap_points.reshape(batch * ears, *heatmap_points.shape[2:])
        if heatmap_points is not None
        else None
    )
    flattened_geodesic = (
        geodesic_distances.reshape(batch * ears, *geodesic_distances.shape[2:])
        if geodesic_distances is not None
        else None
    )
    flattened_votes = (
        vote_offsets.reshape(batch * ears, *vote_offsets.shape[2:])
        if vote_offsets is not None
        else None
    )
    flattened_curve_logits = (
        curve_logits.reshape(batch * ears, *curve_logits.shape[2:])
        if curve_logits is not None
        else None
    )
    flattened_curve_arc = (
        curve_arc_coordinates.reshape(
            batch * ears, *curve_arc_coordinates.shape[2:]
        )
        if curve_arc_coordinates is not None
        else None
    )
    flattened_curve_fractions = (
        curve_landmark_fractions.reshape(
            batch * ears, *curve_landmark_fractions.shape[2:]
        )
        if curve_landmark_fractions is not None
        else None
    )
    return (
        flattened_prediction,
        flattened_target,
        flattened_scale,
        flattened_dense,
        flattened_logits,
        flattened_points,
        flattened_geodesic,
        flattened_votes,
        flattened_curve_logits,
        flattened_curve_arc,
        flattened_curve_fractions,
        int(batch * ears),
    )


def probe_batch_size(
    model: torch.nn.Module,
    sample: Mapping[str, object],
    device: torch.device,
    candidates: Sequence[int] = (32, 16, 8, 4, 2, 1),
    amp: bool = True,
    amp_dtype: str = "auto",
    landmark_loss_weights: Mapping[str, float] | None = None,
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
                points = sample["points"].unsqueeze(0).expand(
                    candidate, *([-1] * sample["points"].ndim)
                ).contiguous().to(device)
            model.zero_grad(set_to_none=True)
            with _autocast(device, amp, amp_dtype):
                if landmark_loss_weights is None:
                    output = model(face_features, neighbors) if "face_features" in sample else model(points)
                    probe_loss = output.float().square().mean()
                else:
                    target = sample["landmarks"].unsqueeze(0).expand(
                        candidate, *([-1] * sample["landmarks"].ndim)
                    ).contiguous().to(device)
                    sample_scale = sample["scale"]
                    scale = (
                        sample_scale.reshape(1).expand(candidate)
                        if sample_scale.ndim == 0
                        else sample_scale.unsqueeze(0).expand(
                            candidate, *([-1] * sample_scale.ndim)
                        )
                    ).contiguous().to(device)
                    dense = sample.get("dense_surface")
                    dense = (
                        dense.unsqueeze(0).expand(
                            candidate, *([-1] * dense.ndim)
                        ).contiguous().to(device)
                        if dense is not None
                        else None
                    )
                    geodesic = sample.get("geodesic_distances_mm")
                    geodesic = (
                        geodesic.unsqueeze(0).expand(
                            candidate, *([-1] * geodesic.ndim)
                        ).contiguous().to(device)
                        if geodesic is not None
                        else None
                    )
                    curve_fractions = sample.get("curve_landmark_fractions")
                    curve_fractions = (
                        curve_fractions.unsqueeze(0).expand(
                            candidate, *([-1] * curve_fractions.ndim)
                        ).contiguous().to(device)
                        if curve_fractions is not None
                        else None
                    )
                    if float(landmark_loss_weights.get("heatmap", 0.0)):
                        if "face_features" in sample:
                            raise ValueError("surface heatmap loss is unavailable for MeshNet")
                        details = model.forward_with_details(points)
                        output = details["final"]
                        heatmap_logits = details.get("heatmap_logits")
                        heatmap_points = details.get("surface_candidates")
                        vote_offsets = details.get("surface_vote_offsets")
                        curve_logits = details.get("curve_logits")
                        curve_arc_coordinates = details.get(
                            "curve_arc_coordinates"
                        )
                        cascade_aux_predictions = details.get(
                            "cascade_aux_predictions"
                        )
                        cascade_aux_logits = details.get("cascade_aux_logits")
                    else:
                        output = model(face_features, neighbors) if "face_features" in sample else model(points)
                        heatmap_logits = None
                        heatmap_points = None
                        vote_offsets = None
                        curve_logits = None
                        curve_arc_coordinates = None
                        cascade_aux_predictions = None
                        cascade_aux_logits = None
                    (
                        output,
                        target,
                        scale,
                        dense,
                        heatmap_logits,
                        heatmap_points,
                        geodesic,
                        vote_offsets,
                        curve_logits,
                        curve_arc_coordinates,
                        curve_fractions,
                        _,
                    ) = _flatten_landmark_batch(
                        output,
                        target,
                        scale,
                        dense,
                        heatmap_logits,
                        heatmap_points,
                        geodesic,
                        vote_offsets,
                        curve_logits,
                        curve_arc_coordinates,
                        curve_fractions,
                    )
                    probe_loss = proposal_landmark_loss(
                        output.float(),
                        target.float(),
                        scale,
                        dense_surface=dense,
                        anchor_weight=float(landmark_loss_weights.get("anchor", 0.0)),
                        spacing_weight=float(landmark_loss_weights.get("spacing", 0.0)),
                        surface_weight=float(landmark_loss_weights.get("surface", 0.0)),
                        heatmap_logits=heatmap_logits,
                        heatmap_surface_points=heatmap_points,
                        heatmap_weight=float(landmark_loss_weights.get("heatmap", 0.0)),
                        heatmap_sigma_mm=float(landmark_loss_weights.get("heatmap_sigma_mm", 2.0)),
                        heatmap_geodesic_distances_mm=geodesic,
                        vote_offsets=vote_offsets,
                        vote_weight=float(landmark_loss_weights.get("vote", 0.0)),
                        vote_radius_mm=float(landmark_loss_weights.get("vote_radius_mm", 6.0)),
                        curve_logits=curve_logits,
                        curve_arc_coordinates=curve_arc_coordinates,
                        curve_landmark_fractions=curve_fractions,
                        curve_weight=float(landmark_loss_weights.get("curve", 0.0)),
                        curve_arc_weight=float(landmark_loss_weights.get("curve_arc", 0.0)),
                        curve_sigma_mm=float(landmark_loss_weights.get("curve_sigma_mm", 3.0)),
                        curve_arc_radius_mm=float(landmark_loss_weights.get("curve_arc_radius_mm", 4.0)),
                        cascade_aux_predictions=cascade_aux_predictions,
                        cascade_aux_logits=cascade_aux_logits,
                        cascade_coordinate_weight=float(landmark_loss_weights.get("cascade_coordinate", 0.0)),
                        cascade_heatmap_weight=float(landmark_loss_weights.get("cascade_heatmap", 0.0)),
                        cascade_heatmap_sigma_mm=float(landmark_loss_weights.get("cascade_heatmap_sigma_mm", 2.0)),
                    )["total"]
                probe_loss.backward()
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


def _landmark_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    encoder_learning_rate: float | None,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, float]:
    """Build AdamW with an optional lower rate for the point backbone."""
    learning_rate = float(learning_rate)
    resolved_encoder_rate = (
        learning_rate
        if encoder_learning_rate is None
        else float(encoder_learning_rate)
    )
    weight_decay = float(weight_decay)
    if learning_rate <= 0.0 or resolved_encoder_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")

    encoder = getattr(model, "encoder", None)
    if encoder is None and encoder_learning_rate is not None:
        raise ValueError(
            "encoder_learning_rate requires a landmark model with an encoder"
        )
    if encoder is None or resolved_encoder_rate == learning_rate:
        optimizer = torch.optim.AdamW(
            [{"params": model.parameters(), "lr": learning_rate, "group_name": "all"}],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        return optimizer, resolved_encoder_rate

    encoder_parameters = list(encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    remaining_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids
    ]
    if not encoder_parameters or not remaining_parameters:
        raise ValueError(
            "separate encoder learning rate requires nonempty encoder and head parameters"
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": resolved_encoder_rate,
                "group_name": "encoder",
            },
            {
                "params": remaining_parameters,
                "lr": learning_rate,
                "group_name": "heads_and_refiner",
            },
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer, resolved_encoder_rate


def _landmark_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    minimum_learning_rate: float,
):
    """Return the legacy cosine schedule or a deterministic warm-up cosine schedule."""
    epochs = int(epochs)
    warmup_epochs = int(warmup_epochs)
    minimum_learning_rate = float(minimum_learning_rate)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if minimum_learning_rate < 0.0:
        raise ValueError("minimum_learning_rate must be non-negative")
    initial_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if any(minimum_learning_rate > rate for rate in initial_rates):
        raise ValueError(
            "minimum_learning_rate cannot exceed an optimizer group's initial rate"
        )

    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=minimum_learning_rate,
        )

    lambdas = []
    for initial_rate in initial_rates:
        minimum_factor = minimum_learning_rate / initial_rate

        def multiplier(
            step: int,
            *,
            minimum_factor: float = minimum_factor,
        ) -> float:
            if step < warmup_epochs:
                if warmup_epochs == 1:
                    warmup_factor = 1.0
                else:
                    warmup_factor = 0.1 + 0.9 * float(step) / float(
                        warmup_epochs - 1
                    )
                return max(minimum_factor, warmup_factor)
            progress = float(step - warmup_epochs) / float(
                max(1, epochs - warmup_epochs)
            )
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
            return minimum_factor + (1.0 - minimum_factor) * cosine

        lambdas.append(multiplier)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def _validate_landmark_resume_config(
    saved: Mapping[str, object], current: Mapping[str, object]
) -> None:
    """Reject optimizer/schedule changes that would corrupt a resumed run."""
    if not saved:
        return
    if saved.get("amp_dtype") != current.get("amp_dtype"):
        raise ValueError(
            "cannot resume landmarks with a different AMP dtype: "
            f"checkpoint={saved.get('amp_dtype')}, current={current.get('amp_dtype')}"
        )

    optimizer_keys = (
        "optimizer",
        "learning_rate",
        "encoder_learning_rate",
        "weight_decay",
        "scheduler",
        "warmup_epochs",
        "warmup_start_factor",
        "minimum_learning_rate",
        "gradient_clip_norm",
        "epochs",
        "patience",
        "effective_batch_size",
        "physical_batch_size",
        "gradient_accumulation",
    )
    if "learning_rate" not in saved:
        legacy_defaults = {
            "learning_rate": 1e-3,
            "encoder_learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_epochs": 0,
            "minimum_learning_rate": 0.0,
            "gradient_clip_norm": 0.0,
        }
        mismatches = [
            key
            for key, expected in legacy_defaults.items()
            if current.get(key) != expected
        ]
        if mismatches:
            raise ValueError(
                "cannot resume a legacy landmark checkpoint with new optimizer "
                f"settings ({', '.join(mismatches)}); use a new output directory"
            )
        return

    mismatches = [
        key for key in optimizer_keys if saved.get(key) != current.get(key)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={saved.get(key)!r}, current={current.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(
            "cannot resume landmarks with different training settings; " + details
        )


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
            model,
            train_dataset[0],
            device,
            amp=amp,
            amp_dtype=amp_dtype,
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
    encoder_learning_rate: float | None = None,
    warmup_epochs: int = 0,
    minimum_learning_rate: float = 0.0,
    gradient_clip_norm: float = 0.0,
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
    ears_per_item = int(getattr(train_dataset, "ears_per_item", 1))
    if ears_per_item not in {1, 2}:
        raise ValueError("landmark dataset ears_per_item must be one or two")
    if batch_size <= 0:
        maximum_items = max(1, int(effective_batch_size) // ears_per_item)
        probe_candidates = tuple(
            candidate
            for candidate in (32, 16, 8, 4, 2, 1)
            if candidate <= maximum_items
        )
        batch_size = probe_batch_size(
            model,
            train_dataset[0],
            device,
            candidates=probe_candidates,
            amp=amp,
            amp_dtype=amp_dtype,
            landmark_loss_weights=loss_weights,
        )
    accumulation = max(
        1,
        int(
            np.ceil(
                effective_batch_size / float(batch_size * ears_per_item)
            )
        ),
    )
    train_loader = _loader(train_dataset, batch_size, workers, True)
    validation_loader = _loader(validation_dataset, batch_size, workers, False) if validation_dataset else None
    optimizer, resolved_encoder_rate = _landmark_optimizer(
        model,
        learning_rate,
        encoder_learning_rate,
        weight_decay,
    )
    scheduler = _landmark_scheduler(
        optimizer,
        epochs,
        warmup_epochs,
        minimum_learning_rate,
    )
    gradient_clip_norm = float(gradient_clip_norm)
    if gradient_clip_norm < 0.0:
        raise ValueError("gradient_clip_norm must be non-negative")
    training_config = {
        **runtime_config,
        "optimizer": "adamw",
        "learning_rate": float(learning_rate),
        "encoder_learning_rate": float(resolved_encoder_rate),
        "weight_decay": float(weight_decay),
        "scheduler": "warmup_cosine" if int(warmup_epochs) else "cosine",
        "warmup_epochs": int(warmup_epochs),
        "warmup_start_factor": 0.1 if int(warmup_epochs) else 1.0,
        "minimum_learning_rate": float(minimum_learning_rate),
        "gradient_clip_norm": gradient_clip_norm,
        "epochs": int(epochs),
        "patience": int(patience),
        "effective_batch_size": int(effective_batch_size),
        "physical_batch_size": int(batch_size),
        "ears_per_item": ears_per_item,
        "physical_ear_batch_size": int(batch_size * ears_per_item),
        "gradient_accumulation": int(accumulation),
    }
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
        _validate_landmark_resume_config(saved_runtime, training_config)
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
        if history:
            best_history_index = min(
                range(len(history)),
                key=lambda index: history[index]["validation"]["mean_distance"],
            )
            stale = len(history) - best_history_index - 1
    if start_epoch > epochs:
        best_item = min(history, key=lambda item: item["validation"]["mean_distance"])
        return {
            "best_md_mm": float(best_item["validation"]["mean_distance"]),
            "best_epoch": int(best_item["epoch"]),
            "physical_batch_size": int(best_item["physical_batch_size"]),
            "gradient_accumulation": int(best_item["gradient_accumulation"]),
            "training_config": training_config,
            **runtime_config,
        }

    def run(loader, training: bool):
        model.train(training)
        if training:
            optimizer.zero_grad(set_to_none=True)
        sums = {
            "total": 0.0,
            "mean_distance": 0.0,
            "anchor": 0.0,
            "spacing": 0.0,
            "surface": 0.0,
            "heatmap": 0.0,
            "vote": 0.0,
            "curve": 0.0,
            "curve_arc": 0.0,
            "cascade_coordinate": 0.0,
            "cascade_heatmap": 0.0,
        }
        count = 0
        for step, batch in enumerate(loader, 1):
            if "face_features" in batch:
                face_features = batch["face_features"].to(device)
                neighbors = batch["neighbors"].to(device)
                batch_count = len(face_features)
            else:
                points = batch["points"].to(device)
                batch_count = len(points) * ears_per_item
            target = batch["landmarks"].to(device)
            dense = batch.get("dense_surface")
            dense = dense.to(device) if dense is not None else None
            geodesic = batch.get("geodesic_distances_mm")
            geodesic = geodesic.to(device) if geodesic is not None else None
            curve_fractions = batch.get("curve_landmark_fractions")
            curve_fractions = (
                curve_fractions.to(device)
                if curve_fractions is not None
                else None
            )
            with torch.set_grad_enabled(training):
                with _autocast(device, amp, amp_dtype):
                    if float(loss_weights.get("heatmap", 0.0)):
                        if "face_features" in batch:
                            raise ValueError(
                                "surface heatmap loss is unavailable for MeshNet"
                            )
                        details = model.forward_with_details(points)
                        prediction = details["final"]
                        heatmap_logits = details.get("heatmap_logits")
                        heatmap_points = details.get("surface_candidates")
                        vote_offsets = details.get("surface_vote_offsets")
                        curve_logits = details.get("curve_logits")
                        curve_arc_coordinates = details.get(
                            "curve_arc_coordinates"
                        )
                        cascade_aux_predictions = details.get(
                            "cascade_aux_predictions"
                        )
                        cascade_aux_logits = details.get("cascade_aux_logits")
                    else:
                        prediction = model(face_features, neighbors) if "face_features" in batch else model(points)
                        heatmap_logits = None
                        heatmap_points = None
                        vote_offsets = None
                        curve_logits = None
                        curve_arc_coordinates = None
                        cascade_aux_predictions = None
                        cascade_aux_logits = None
                    (
                        prediction,
                        target,
                        scale,
                        dense,
                        heatmap_logits,
                        heatmap_points,
                        geodesic,
                        vote_offsets,
                        curve_logits,
                        curve_arc_coordinates,
                        curve_fractions,
                        flattened_count,
                    ) = _flatten_landmark_batch(
                        prediction,
                        target,
                        batch["scale"].to(device),
                        dense,
                        heatmap_logits,
                        heatmap_points,
                        geodesic,
                        vote_offsets,
                        curve_logits,
                        curve_arc_coordinates,
                        curve_fractions,
                    )
                    if flattened_count != batch_count:
                        raise RuntimeError("landmark batch ear count is inconsistent")
                    losses = proposal_landmark_loss(
                        prediction.float(), target.float(), scale, dense_surface=dense,
                        anchor_weight=float(loss_weights.get("anchor", 0.0)),
                        spacing_weight=float(loss_weights.get("spacing", 0.0)),
                        surface_weight=float(loss_weights.get("surface", 0.0)),
                        heatmap_logits=heatmap_logits,
                        heatmap_surface_points=heatmap_points,
                        heatmap_weight=float(loss_weights.get("heatmap", 0.0)),
                        heatmap_sigma_mm=float(loss_weights.get("heatmap_sigma_mm", 2.0)),
                        heatmap_geodesic_distances_mm=geodesic,
                        vote_offsets=vote_offsets,
                        vote_weight=float(loss_weights.get("vote", 0.0)),
                        vote_radius_mm=float(loss_weights.get("vote_radius_mm", 6.0)),
                        curve_logits=curve_logits,
                        curve_arc_coordinates=curve_arc_coordinates,
                        curve_landmark_fractions=curve_fractions,
                        curve_weight=float(loss_weights.get("curve", 0.0)),
                        curve_arc_weight=float(loss_weights.get("curve_arc", 0.0)),
                        curve_sigma_mm=float(loss_weights.get("curve_sigma_mm", 3.0)),
                        curve_arc_radius_mm=float(loss_weights.get("curve_arc_radius_mm", 4.0)),
                        cascade_aux_predictions=cascade_aux_predictions,
                        cascade_aux_logits=cascade_aux_logits,
                        cascade_coordinate_weight=float(loss_weights.get("cascade_coordinate", 0.0)),
                        cascade_heatmap_weight=float(loss_weights.get("cascade_heatmap", 0.0)),
                        cascade_heatmap_sigma_mm=float(loss_weights.get("cascade_heatmap_sigma_mm", 2.0)),
                    )
                if training:
                    scaler.scale(losses["total"] / accumulation).backward()
                    if step % accumulation == 0 or step == len(loader):
                        if gradient_clip_norm > 0.0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), gradient_clip_norm
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
            for key in sums:
                sums[key] += float(losses[key].detach()) * batch_count
            count += batch_count
        return {key: value / max(count, 1) for key, value in sums.items()}

    for epoch in range(start_epoch, epochs + 1):
        epoch_learning_rates = {
            str(group.get("group_name", f"group_{index}")): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
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
            "physical_ear_batch_size": int(batch_size * ears_per_item),
            "ears_per_item": ears_per_item,
            "gradient_accumulation": accumulation,
            "learning_rates": epoch_learning_rates,
            "training_config": training_config,
            **runtime_config,
        }
        history.append({"epoch": epoch, **metrics})
        _save_component_checkpoint(last_path, "landmarks", model, model_config, optimizer, scheduler, epoch, metrics, data_config, training_config, scaler)
        if improved:
            _save_component_checkpoint(output / "best_landmarks.pt", "landmarks", model, model_config, optimizer, scheduler, epoch, metrics, data_config, training_config, scaler)
        _json_dump(
            metrics_path,
            {"history": history, "best": best, "training_config": training_config},
        )
        if validation_loader and stale >= patience:
            break
    best_item = min(history, key=lambda item: item["validation"]["mean_distance"])
    return {
        "best_md_mm": best,
        "best_epoch": best_item["epoch"],
        "physical_batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "training_config": training_config,
        **runtime_config,
    }


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
