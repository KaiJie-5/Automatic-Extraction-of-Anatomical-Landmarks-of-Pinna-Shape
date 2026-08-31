"""Explicit mixed-precision policy shared by training and inference."""

from __future__ import annotations

import contextlib
from typing import Mapping

import torch


AMP_DTYPE_NAMES = ("auto", "float16", "bfloat16", "none")


def normalize_amp_dtype(value: str | None) -> str:
    name = "auto" if value is None else str(value).lower()
    aliases = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "off": "none",
        "disabled": "none",
    }
    name = aliases.get(name, name)
    if name not in AMP_DTYPE_NAMES:
        raise ValueError(
            f"unsupported AMP dtype {value!r}; expected one of {AMP_DTYPE_NAMES}"
        )
    return name


def resolve_amp_dtype(
    device: torch.device,
    enabled: bool,
    requested: str | None = "auto",
) -> torch.dtype | None:
    """Resolve a requested AMP policy without silently changing explicit dtypes."""

    name = normalize_amp_dtype(requested)
    if not enabled or name == "none" or device.type != "cuda":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 AMP was requested but the CUDA device lacks support")
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def resolved_amp_dtype_name(
    device: torch.device,
    enabled: bool,
    requested: str | None = "auto",
) -> str:
    dtype = resolve_amp_dtype(device, enabled, requested)
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    return "none"


def autocast_context(
    device: torch.device,
    enabled: bool,
    requested: str | None = "auto",
):
    dtype = resolve_amp_dtype(device, enabled, requested)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def grad_scaler_enabled(
    device: torch.device,
    enabled: bool,
    requested: str | None = "auto",
) -> bool:
    return resolve_amp_dtype(device, enabled, requested) is torch.float16


def checkpoint_amp_dtype(model_config: Mapping[str, object]) -> str:
    """Return the recorded inference dtype, or no autocast for old checkpoints."""

    return normalize_amp_dtype(model_config.get("amp_dtype", "none"))


def checkpoint_autocast_context(
    device: torch.device,
    model_config: Mapping[str, object],
):
    requested = checkpoint_amp_dtype(model_config)
    return autocast_context(device, requested != "none", requested)
