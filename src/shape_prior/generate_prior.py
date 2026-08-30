"""Generate a standalone PCA prior artifact for later inference integration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..dataset import Dataset
from .fitting import (
    file_sha256,
    load_center_predictions,
    read_json,
    select_training_subjects,
    build_training_shapes,
)
from .pca import COORDINATE_FRAME, NORMALIZATION, PCAShapePrior


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--outer-fold", required=True, help="fold index or final")
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--components", type=positive_int, default=32)
    parser.add_argument("--beta", type=unit_float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = Dataset(args.mesh_dir, args.landmarks_dir)
    folds = read_json(args.folds_json)
    calibration = read_json(args.calibration_json)
    dataset_ids = [dataset.get_identifier(index) for index in range(len(dataset))]
    predictions = load_center_predictions(args.predictions_json, dataset_ids)
    subject_ids = select_training_subjects(dataset, folds, args.outer_fold)
    dataset_checksum = hashlib.sha256(
        "\n".join(sorted(dataset_ids)).encode("utf-8")
    ).hexdigest()
    if folds.get("subject_checksum") != dataset_checksum:
        raise ValueError("dataset subject IDs do not match folds JSON checksum")
    shapes = build_training_shapes(dataset, subject_ids, predictions, calibration)

    prior = PCAShapePrior.fit(shapes, n_components=args.components, beta=args.beta)
    prior.save(args.output)

    manifest = {
        "mode": "final" if args.outer_fold == "final" else "fold",
        "outer_fold": args.outer_fold,
        "subject_count": len(subject_ids),
        "ear_count": int(shapes.shape[0]),
        "components": prior.n_components,
        "beta": prior.beta,
        "output": str(Path(args.output)),
        "coordinate_frame": COORDINATE_FRAME,
        "normalization": NORMALIZATION,
        "train_subject_ids": subject_ids,
        "folds_json_sha256": file_sha256(args.folds_json),
        "predictions_json_sha256": file_sha256(args.predictions_json),
        "calibration_json_sha256": file_sha256(args.calibration_json),
        "prior_sha256": file_sha256(args.output),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Wrote PCA prior for {manifest['mode']} {args.outer_fold}: "
        f"{manifest['subject_count']} subjects, {manifest['ear_count']} ears, "
        f"{prior.n_components} components, beta={prior.beta:g}"
    )
    print(f"Prior: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
