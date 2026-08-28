"""Embed a generated PCA shape prior into a final pipeline checkpoint copy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .pca import PCAShapePrior


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _prior_dict(prior: PCAShapePrior, beta: float) -> dict:
    return {
        "mean_shape": prior.mean_shape.astype("float32").tolist(),
        "components": prior.components.astype("float32").tolist(),
        "n_components": int(prior.n_components),
        "beta": float(beta),
        "landmark_count": int(prior.landmark_count),
        "coordinate_frame": prior.coordinate_frame,
        "normalization": prior.normalization,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = _torch_load(args.checkpoint)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    prior = PCAShapePrior.load(args.prior)

    output_checkpoint = dict(checkpoint)
    postprocess = dict(output_checkpoint.get("postprocess", {}))
    postprocess["pca_shape_prior"] = {
        "enabled": True,
        "beta": float(args.beta),
        "prior": _prior_dict(prior, float(args.beta)),
    }
    output_checkpoint["postprocess"] = postprocess

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output)
    print(f"Input checkpoint: {args.checkpoint}")
    print(f"PCA prior: {args.prior}")
    print(f"Output checkpoint: {args.output}")
    print(f"PCA beta: {float(args.beta):g}")
    print(f"PCA components: {prior.n_components}")


if __name__ == "__main__":
    main()
