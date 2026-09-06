"""Generate a leakage-safe bilateral mean/asymmetry PCA prior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..dataset import Dataset
from .bilateral_pca import (
    BILATERAL_NORMALIZATION,
    BilateralMeanAsymmetryPCAPrior,
)
from .fitting import (
    build_training_shape_pairs,
    file_sha256,
    load_center_predictions,
    read_json,
    select_training_subjects,
)
from .generate_prior import positive_int, unit_float
from .pca import COORDINATE_FRAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--landmarks-dir", required=True)
    parser.add_argument("--folds-json", required=True)
    parser.add_argument("--outer-fold", required=True, help="fold index or final")
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--common-components", type=positive_int, default=64)
    parser.add_argument("--asymmetry-components", type=positive_int, default=32)
    parser.add_argument("--common-beta", type=unit_float, default=1.0)
    parser.add_argument("--asymmetry-beta", type=unit_float, default=1.0)
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
    pairs = build_training_shape_pairs(
        dataset, subject_ids, predictions, calibration
    )
    prior = BilateralMeanAsymmetryPCAPrior.fit(
        pairs,
        common_components=args.common_components,
        asymmetry_components=args.asymmetry_components,
        common_beta=args.common_beta,
        asymmetry_beta=args.asymmetry_beta,
    )
    prior.save(args.output)

    manifest = {
        "schema_version": 1,
        "prior_type": "bilateral_mean_asymmetry_pca",
        "mode": "final" if args.outer_fold == "final" else "fold",
        "outer_fold": args.outer_fold,
        "subject_count": len(subject_ids),
        "pair_count": int(pairs.shape[0]),
        "ear_count": int(pairs.shape[0] * pairs.shape[1]),
        "common_components": prior.common_n_components,
        "asymmetry_components": prior.asymmetry_n_components,
        "common_beta": prior.common_beta,
        "asymmetry_beta": prior.asymmetry_beta,
        "output": str(Path(args.output)),
        "coordinate_frame": COORDINATE_FRAME,
        "normalization": BILATERAL_NORMALIZATION,
        "ear_order": ["left", "mirrored_right"],
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
        "Wrote bilateral mean/asymmetry PCA prior for "
        f"{manifest['mode']} {args.outer_fold}: "
        f"{manifest['subject_count']} subject pairs, "
        f"common={prior.common_n_components}, "
        f"asymmetry={prior.asymmetry_n_components} components"
    )
    print(f"Prior: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
