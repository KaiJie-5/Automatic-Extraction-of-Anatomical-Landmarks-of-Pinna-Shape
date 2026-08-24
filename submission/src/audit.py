"""Strict, consolidated validation for the pinna challenge dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh

from .dataset import Dataset


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_report(path: Path) -> Dict[str, Any]:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("file did not load as a Trimesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh must contain finite vertices with shape (N, 3)")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh contains non-finite vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("mesh must contain triangular faces with shape (F, 3)")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("mesh contains out-of-range face indices")
    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    degenerate = int(np.count_nonzero(double_area <= 1e-10))
    if degenerate:
        raise ValueError(f"mesh contains {degenerate} degenerate faces")
    try:
        components = int(mesh.body_count)
    except Exception:
        components = -1
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
        "extents": np.ptp(vertices, axis=0).tolist(),
        "surface_area": float(double_area.sum() * 0.5),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "components": components,
        "has_vertex_normals": bool(np.asarray(mesh.vertex_normals).shape == vertices.shape),
        "sha256": _file_sha256(path),
    }


def audit_dataset(
    mesh_dir: str,
    landmarks_dir: str,
    output_root: Optional[str] = None,
    fail_on_fatal: bool = True,
    expected_subjects: Optional[int] = None,
) -> Dict[str, Any]:
    """Audit every discovered subject and optionally persist a versioned report.

    Reports are always written before a fatal-data exception is raised.
    """
    mesh_path = Path(mesh_dir)
    landmark_path = Path(landmarks_dir)
    mesh_ids = {path.stem for path in mesh_path.glob("*.ply")}
    left_suffix = "_left_ear_landmarks"
    right_suffix = "_right_ear_landmarks"
    left_ids = {
        path.stem[: -len(left_suffix)]
        for path in landmark_path.glob(f"*{left_suffix}.csv")
    }
    right_ids = {
        path.stem[: -len(right_suffix)]
        for path in landmark_path.glob(f"*{right_suffix}.csv")
    }
    subject_ids = sorted(mesh_ids | left_ids | right_ids)
    subjects: Dict[str, Any] = {}
    fatal_count = 0
    warning_count = 0

    for subject_id in subject_ids:
        fatal = []
        warnings = []
        record: Dict[str, Any] = {"fatal": fatal, "warnings": warnings}
        files = {
            "mesh": mesh_path / f"{subject_id}.ply",
            "left": landmark_path / f"{subject_id}{left_suffix}.csv",
            "right": landmark_path / f"{subject_id}{right_suffix}.csv",
        }
        missing = [name for name, path in files.items() if not path.exists()]
        if missing:
            fatal.append(f"missing files: {', '.join(missing)}")
        if files["mesh"].exists():
            try:
                record["mesh"] = _mesh_report(files["mesh"])
            except Exception as exc:
                fatal.append(f"mesh: {exc}")
        for ear in ("left", "right"):
            if files[ear].exists():
                try:
                    points = Dataset._load_landmarks(files[ear])
                    record[ear] = {
                        "rows": int(len(points)),
                        "minimum": points.min(axis=0).tolist(),
                        "maximum": points.max(axis=0).tolist(),
                        "sha256": _file_sha256(files[ear]),
                    }
                except Exception as exc:
                    fatal.append(f"{ear} landmarks: {exc}")

        mesh_info = record.get("mesh")
        if mesh_info:
            if not mesh_info["watertight"]:
                warnings.append("mesh is not watertight")
            if not mesh_info["winding_consistent"]:
                warnings.append("mesh winding is inconsistent")
            if mesh_info["components"] > 1:
                warnings.append(f"mesh has {mesh_info['components']} connected bodies")
        subjects[subject_id] = record

    complete_meshes = {
        subject_id: record["mesh"]
        for subject_id, record in subjects.items()
        if not record["fatal"] and "mesh" in record
    }
    if complete_meshes:
        outlier_metrics = {
            "vertex count": {
                subject_id: np.log1p(info["vertices"])
                for subject_id, info in complete_meshes.items()
            },
            "Y extent": {
                subject_id: float(info["extents"][1])
                for subject_id, info in complete_meshes.items()
            },
            "surface density": {
                subject_id: float(info["vertices"]) / max(float(info["surface_area"]), 1e-8)
                for subject_id, info in complete_meshes.items()
            },
        }
        for metric_name, values_by_subject in outlier_metrics.items():
            values = np.asarray(list(values_by_subject.values()), dtype=np.float64)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad <= 1e-12:
                continue
            for subject_id, value in values_by_subject.items():
                robust_z = 0.6745 * abs(float(value) - median) / mad
                if robust_z > 5.0:
                    subjects[subject_id]["warnings"].append(
                        f"unusual {metric_name} (robust z={robust_z:.2f})"
                    )

    fatal_count = sum(len(record["fatal"]) for record in subjects.values())
    warning_count = sum(len(record["warnings"]) for record in subjects.values())

    complete = [
        subject_id for subject_id, record in subjects.items() if not record["fatal"]
    ]
    global_fatal = []
    if expected_subjects is not None and len(complete) != int(expected_subjects):
        global_fatal.append(
            f"expected {int(expected_subjects)} complete subjects, found {len(complete)}"
        )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_dir": str(mesh_path.resolve()),
        "landmarks_dir": str(landmark_path.resolve()),
        "summary": {
            "discovered_subjects": len(subject_ids),
            "complete_subjects": len(complete),
            "complete_ears": len(complete) * 2,
            "fatal_issues": fatal_count + len(global_fatal),
            "warnings": warning_count,
            "p0027_included": "P0027" in complete,
            "kemar_included": "KEMAR" in complete,
        },
        "global_fatal": global_fatal,
        "subjects": subjects,
    }

    if output_root:
        stamp = datetime.now(timezone.utc).strftime("audit_%Y%m%dT%H%M%SZ")
        output_dir = Path(output_root) / stamp
        output_dir.mkdir(parents=True, exist_ok=False)
        with (output_dir / "dataset_audit.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        with (output_dir / "dataset_audit.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "subject_id",
                    "fatal",
                    "warnings",
                    "vertices",
                    "faces",
                    "y_extent",
                    "watertight",
                    "components",
                ]
            )
            for subject_id, record in subjects.items():
                info = record.get("mesh", {})
                writer.writerow(
                    [
                        subject_id,
                        " | ".join(record["fatal"]),
                        " | ".join(record["warnings"]),
                        info.get("vertices", ""),
                        info.get("faces", ""),
                        (info.get("extents") or ["", "", ""])[1],
                        info.get("watertight", ""),
                        info.get("components", ""),
                    ]
                )
        report["output_dir"] = str(output_dir)

    total_fatal = fatal_count + len(global_fatal)
    if total_fatal and fail_on_fatal:
        location = report.get("output_dir", "the returned audit report")
        raise ValueError(f"Dataset audit found {total_fatal} fatal issue(s); see {location}")
    return report
