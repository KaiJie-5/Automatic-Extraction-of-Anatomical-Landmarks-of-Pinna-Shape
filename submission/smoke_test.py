"""End-to-end deterministic smoke test for the packaged challenge estimator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.dataset import Dataset
from src.estimator import LandmarkExtractor
from src.metrics import compute_mean_landmark_distance


def _validate_output(name: str, values: np.ndarray) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} output must be a numpy.ndarray")
    if values.shape != (85, 3):
        raise ValueError(f"{name} output has shape {values.shape}, expected (85, 3)")
    if values.dtype != np.float32:
        raise TypeError(f"{name} output has dtype {values.dtype}, expected float32")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} output contains NaN or infinity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/final_pipeline.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    dataset = Dataset(
        mesh_dir=str(data_root / "mesh"),
        landmarks_dir=str(data_root / "landmarks"),
    )
    if len(dataset) != 1 or dataset.get_identifier(0) != "KEMAR":
        raise RuntimeError("The packaged smoke data must contain exactly the KEMAR sample")
    mesh, left_target, right_target = dataset[0]

    extractor = LandmarkExtractor(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        device=args.device,
    )
    started = time.perf_counter()
    first_left, first_right = extractor.extract(mesh)
    if extractor.device.type == "cuda":
        import torch

        torch.cuda.synchronize(extractor.device)
    first_seconds = time.perf_counter() - started
    _validate_output("left", first_left)
    _validate_output("right", first_right)

    second_left, second_right = extractor.extract(mesh)
    _validate_output("repeat left", second_left)
    _validate_output("repeat right", second_right)
    if not np.array_equal(first_left, second_left) or not np.array_equal(
        first_right, second_right
    ):
        raise RuntimeError("Repeated inference was not bit-for-bit deterministic")

    report = {
        "checkpoint": str(extractor.checkpoint_path),
        "device": str(extractor.device),
        "left_dtype": str(first_left.dtype),
        "left_md_mm_on_training_smoke_sample": float(
            compute_mean_landmark_distance(first_left, left_target)
        ),
        "output_shape": list(first_left.shape),
        "repeat_bitwise_equal": True,
        "right_dtype": str(first_right.dtype),
        "right_md_mm_on_training_smoke_sample": float(
            compute_mean_landmark_distance(first_right, right_target)
        ),
        "seconds_first_full_head": first_seconds,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
