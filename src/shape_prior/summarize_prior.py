"""Summarize the locked five-fold/three-seed PCA projection experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


FOLDS = tuple(range(5))
DEFAULT_SEEDS = (42, 43, 44)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ear_values(report: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    raw = []
    candidate = []
    subjects = report.get("subject_level_raw_and_pca_errors", {})
    if not isinstance(subjects, Mapping):
        raise ValueError("PCA report is missing subject-level ear metrics")
    for subject_id in sorted(subjects):
        ears = subjects[subject_id]
        if not isinstance(ears, Mapping) or set(ears) != {"left", "right"}:
            raise ValueError(f"invalid ear metrics for subject {subject_id}")
        for ear in ("left", "right"):
            row = ears[ear]
            raw.append(float(row["projected_md_mm"]))
            candidate.append(float(row["pca_projected_md_mm"]))
    raw_array = np.asarray(raw, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    if (
        raw_array.ndim != 1
        or not len(raw_array)
        or raw_array.shape != candidate_array.shape
        or not np.isfinite(raw_array).all()
        or not np.isfinite(candidate_array).all()
    ):
        raise ValueError("PCA report contains invalid ear metrics")
    return raw_array, candidate_array


def _distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def summarize(report_root: str | Path, seeds: Sequence[int]) -> dict:
    root = Path(report_root)
    expected = {(fold, int(seed)) for fold in FOLDS for seed in seeds}
    configurations = set()
    fold_values = {fold: [[], []] for fold in FOLDS}
    all_raw = []
    all_candidate = []
    reports = []

    for fold, seed in sorted(expected):
        path = root / f"fold{fold}_seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing PCA confirmation report: {path}")
        report = _read_json(path)
        if (
            report.get("schema_version") != 2
            or report.get("component") != "fold_projection_aware_pca_evaluation"
            or int(report.get("outer_fold", -1)) != fold
            or int(report.get("run_seed", -1)) != seed
        ):
            raise ValueError(f"PCA report identity does not match its path: {path}")
        configurations.add((int(report["components"]), float(report["beta"])))
        raw, candidate = _ear_values(report)
        if int(report.get("ear_count", -1)) != len(raw):
            raise ValueError(f"ear count mismatch in {path}")
        if not np.isclose(float(report["projected_mean_md_mm"]), raw.mean(), atol=1e-6):
            raise ValueError(f"baseline mean mismatch in {path}")
        if not np.isclose(
            float(report["pca_projected_mean_md_mm"]),
            candidate.mean(),
            atol=1e-6,
        ):
            raise ValueError(f"candidate mean mismatch in {path}")
        fold_values[fold][0].append(raw)
        fold_values[fold][1].append(candidate)
        all_raw.append(raw)
        all_candidate.append(candidate)
        reports.append(str(path))

    if len(configurations) != 1:
        raise ValueError(
            "all 15 PCA reports must use exactly one locked components/beta setting"
        )
    components, beta = next(iter(configurations))
    raw_pooled = np.concatenate(all_raw)
    candidate_pooled = np.concatenate(all_candidate)
    per_fold = {}
    improved_folds = 0
    for fold in FOLDS:
        raw = np.concatenate(fold_values[fold][0])
        candidate = np.concatenate(fold_values[fold][1])
        improvement = float(raw.mean() - candidate.mean())
        improved_folds += int(improvement > 0.0)
        per_fold[str(fold)] = {
            "baseline_projected_md_mm": float(raw.mean()),
            "pca_projected_md_mm": float(candidate.mean()),
            "improvement_mm": improvement,
            "ear_evaluations": int(len(raw)),
        }

    pooled_improvement = float(raw_pooled.mean() - candidate_pooled.mean())
    return {
        "schema_version": 1,
        "component": "projection_aware_pca_confirmation_summary",
        "components": components,
        "beta": beta,
        "folds": list(FOLDS),
        "seeds": [int(seed) for seed in seeds],
        "report_count": len(reports),
        "ear_evaluations": int(len(raw_pooled)),
        "baseline_projected_distribution_mm": _distribution(raw_pooled),
        "pca_projected_distribution_mm": _distribution(candidate_pooled),
        "pooled_improvement_mm": pooled_improvement,
        "improved_folds": int(improved_folds),
        "per_fold": per_fold,
        "promotion_rule": {
            "lower_pooled_15_run_md": bool(pooled_improvement > 0.0),
            "improve_at_least_three_folds": bool(improved_folds >= 3),
            "promoted": bool(pooled_improvement > 0.0 and improved_folds >= 3),
        },
        "reports": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = summarize(args.report_root, args.seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
