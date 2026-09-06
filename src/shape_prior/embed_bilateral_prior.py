"""Embed a promoted bilateral PCA prior into a copy of a final v2 bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .bilateral_pca import (
    CONTOUR_RANGES,
    BilateralMeanAsymmetryPCAPrior,
    normalise_contour_gate,
)
from .embed_prior import _torch_load


def _prior_dict(
    prior: BilateralMeanAsymmetryPCAPrior,
    common_components: int,
    asymmetry_components: int,
) -> dict:
    return {
        "common_mean": prior.common_mean.astype("float32").tolist(),
        "common_components": prior.common_components[:common_components]
        .astype("float32")
        .tolist(),
        "common_n_components": int(common_components),
        "asymmetry_mean": prior.asymmetry_mean.astype("float32").tolist(),
        "asymmetry_components": prior.asymmetry_components[:asymmetry_components]
        .astype("float32")
        .tolist(),
        "asymmetry_n_components": int(asymmetry_components),
        "landmark_count": int(prior.landmark_count),
        "coordinate_frame": prior.coordinate_frame,
        "normalization": prior.normalization,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--common-components", type=int, required=True)
    parser.add_argument("--asymmetry-components", type=int, required=True)
    parser.add_argument("--common-beta", type=float, required=True)
    parser.add_argument("--asymmetry-beta", type=float, required=True)
    parser.add_argument(
        "--contour-gate",
        nargs="+",
        choices=tuple(CONTOUR_RANGES),
        help=(
            "retain the existing independent PCA outside these contours and "
            "use bilateral PCA inside them"
        ),
    )
    parser.add_argument(
        "--replace-independent-pca",
        action="store_true",
        help="remove an enabled independent-ear PCA prior from the output copy",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = _torch_load(args.checkpoint)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 2:
        raise ValueError("checkpoint must be a complete schema-v2 pipeline bundle")
    prior = BilateralMeanAsymmetryPCAPrior.load(args.prior)
    common_components = int(args.common_components)
    asymmetry_components = int(args.asymmetry_components)
    if not 0 < common_components <= len(prior.common_components):
        raise ValueError("common component count exceeds the saved prior")
    if not 0 < asymmetry_components <= len(prior.asymmetry_components):
        raise ValueError("asymmetry component count exceeds the saved prior")
    common_beta = float(args.common_beta)
    asymmetry_beta = float(args.asymmetry_beta)
    contour_gate = normalise_contour_gate(args.contour_gate)
    if not 0.0 <= common_beta <= 1.0 or not 0.0 <= asymmetry_beta <= 1.0:
        raise ValueError("bilateral PCA blend strengths must be in [0, 1]")

    output_checkpoint = dict(checkpoint)
    postprocess = dict(output_checkpoint.get("postprocess", {}))
    independent = postprocess.get("pca_shape_prior")
    if isinstance(independent, dict) and bool(independent.get("enabled", False)):
        if contour_gate and args.replace_independent_pca:
            raise ValueError(
                "--contour-gate cannot be combined with "
                "--replace-independent-pca"
            )
        if not contour_gate and not args.replace_independent_pca:
            raise ValueError(
                "checkpoint already enables independent PCA; pass "
                "--replace-independent-pca to replace it, or pass "
                "--contour-gate to retain it outside selected contours"
            )
        if args.replace_independent_pca:
            postprocess.pop("pca_shape_prior", None)
    elif contour_gate:
        raise ValueError(
            "--contour-gate requires an enabled independent PCA prior in "
            "the input checkpoint"
        )
    postprocess["bilateral_mean_asymmetry_pca_prior"] = {
        "enabled": True,
        "common_beta": common_beta,
        "asymmetry_beta": asymmetry_beta,
        "ear_order": ["left", "mirrored_right"],
        "contour_gate": list(contour_gate),
        "prior": _prior_dict(
            prior, common_components, asymmetry_components
        ),
    }
    output_checkpoint["postprocess"] = postprocess
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output)
    print(f"Input checkpoint: {args.checkpoint}")
    print(f"Bilateral PCA prior: {args.prior}")
    print(f"Output checkpoint: {args.output}")
    print(
        f"Setting: common={common_components}, beta={common_beta:g}; "
        f"asymmetry={asymmetry_components}, beta={asymmetry_beta:g}"
    )
    if contour_gate:
        print("Bilateral contour gate: " + ", ".join(contour_gate))


if __name__ == "__main__":
    main()
