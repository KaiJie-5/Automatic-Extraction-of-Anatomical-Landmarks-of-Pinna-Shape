"""Leakage-safe subject fold generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def _assign_stratified(
    subject_ids: Sequence[str],
    y_extents: Mapping[str, float],
    vertex_counts: Mapping[str, int],
    n_folds: int,
    seed: int,
) -> List[List[str]]:
    if n_folds < 2 or len(subject_ids) < n_folds:
        raise ValueError("n_folds must be at least 2 and no larger than the subject count")
    ids = sorted(subject_ids)
    y = np.asarray([float(y_extents[item]) for item in ids], dtype=np.float64)
    quantiles = np.quantile(y, np.linspace(0.0, 1.0, n_folds + 1)[1:-1])
    bins = np.digitize(y, quantiles, right=True)
    target_sizes = [len(ids) // n_folds + (1 if idx < len(ids) % n_folds else 0) for idx in range(n_folds)]
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    rng = np.random.default_rng(seed)

    for bin_index in range(n_folds):
        members = [item for item, value in zip(ids, bins) if value == bin_index]
        members.sort(key=lambda item: (np.log1p(vertex_counts[item]), item))
        if bin_index % 2:
            members.reverse()
        offset = int(rng.integers(0, n_folds))
        for rank, subject_id in enumerate(members):
            preferences = [(offset + rank + step) % n_folds for step in range(n_folds)]
            eligible = [idx for idx in preferences if len(folds[idx]) < target_sizes[idx]]
            chosen = min(eligible, key=lambda idx: (len(folds[idx]), preferences.index(idx)))
            folds[chosen].append(subject_id)
    for fold in folds:
        fold.sort()
    if sorted(item for fold in folds for item in fold) != ids:
        raise RuntimeError("fold assignment did not preserve every subject exactly once")
    return folds


def _checksum(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_nested_folds(
    subject_metrics: Mapping[str, Mapping[str, float]],
    seed: int = 42,
    outer_folds: int = 5,
    inner_folds: int = 4,
) -> Dict[str, object]:
    subject_ids = sorted(subject_metrics)
    y_extents = {item: float(subject_metrics[item]["y_extent"]) for item in subject_ids}
    vertex_counts = {item: int(subject_metrics[item]["vertices"]) for item in subject_ids}
    outer = _assign_stratified(subject_ids, y_extents, vertex_counts, outer_folds, seed)
    outer_records = []
    for outer_index, validation in enumerate(outer):
        training = sorted(set(subject_ids) - set(validation))
        inner = _assign_stratified(
            training,
            y_extents,
            vertex_counts,
            inner_folds,
            seed + 1000 + outer_index,
        )
        outer_records.append(
            {
                "fold": outer_index,
                "train": training,
                "validation": validation,
                "inner": [
                    {
                        "fold": inner_index,
                        "train": sorted(set(training) - set(inner_validation)),
                        "validation": inner_validation,
                    }
                    for inner_index, inner_validation in enumerate(inner)
                ],
            }
        )
    return {
        "schema_version": 1,
        "seed": int(seed),
        "subject_count": len(subject_ids),
        "subject_checksum": _checksum(subject_ids),
        "outer": outer_records,
    }


def folds_from_audit(audit_report: Mapping[str, object], seed: int = 42) -> Dict[str, object]:
    metrics = {}
    for subject_id, record in audit_report["subjects"].items():
        if record["fatal"]:
            continue
        mesh = record["mesh"]
        metrics[subject_id] = {"y_extent": mesh["extents"][1], "vertices": mesh["vertices"]}
    return make_nested_folds(metrics, seed=seed)


def save_folds(folds: Mapping[str, object], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(folds, handle, indent=2, sort_keys=True)
