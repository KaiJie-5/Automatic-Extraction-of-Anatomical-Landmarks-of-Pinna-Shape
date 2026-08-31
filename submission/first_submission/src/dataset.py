from typing import Iterable, Optional, Tuple
import numpy as np
import trimesh
from trimesh import Trimesh
import csv
from pathlib import Path


DEFAULT_EXCLUDED_SUBJECT_IDS = frozenset()


class Dataset:
    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        exclude_subject_ids: Optional[Iterable[str]] = None,
    ):
        self.mesh_dir = Path(mesh_dir)
        self.landmarks_dir = Path(landmarks_dir)
        excluded_subject_ids = set(DEFAULT_EXCLUDED_SUBJECT_IDS)
        if exclude_subject_ids is not None:
            excluded_subject_ids.update(exclude_subject_ids)
        self.subject_ids = sorted(
            f.stem for f in self.mesh_dir.glob("*.ply") if f.stem not in excluded_subject_ids
        )

    def __len__(self) -> int:
        return len(self.subject_ids)
    
    def get_identifier(self, idx: int) -> str:
        return self.subject_ids[idx]
    
    def __getitem__(self, idx: int) -> Tuple[Trimesh, np.ndarray, np.ndarray]:
        subject_id = self.subject_ids[idx]

        # --- Load mesh ---
        mesh_path = self.mesh_dir / f"{subject_id}.ply"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, Trimesh):
            raise ValueError(f"{mesh_path} did not load as a triangular mesh")

        # --- Load landmarks ---
        left_path = self.landmarks_dir / f"{subject_id}_left_ear_landmarks.csv"
        right_path = self.landmarks_dir/ f"{subject_id}_right_ear_landmarks.csv"
        landmarks_left = self._load_landmarks(left_path)
        landmarks_right = self._load_landmarks(right_path)

        return mesh, landmarks_left, landmarks_right    
    
    @staticmethod
    def _load_landmarks(filepath: Path) -> np.ndarray:
        """
        input CSV with format: index, [x y z]
        output N x 3 array
        """
        with open(filepath, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        if len(rows) != 85:
            raise ValueError(f"{filepath} must contain exactly 85 rows; found {len(rows)}")

        coords = []
        for expected_index, row in enumerate(rows):
            if len(row) != 2:
                raise ValueError(f"{filepath}:{expected_index + 1} must have exactly 2 columns")
            try:
                actual_index = int(row[0].strip())
            except ValueError as exc:
                raise ValueError(f"{filepath}:{expected_index + 1} has an invalid index") from exc
            if actual_index != expected_index:
                raise ValueError(
                    f"{filepath}:{expected_index + 1} expected index {expected_index}, "
                    f"found {actual_index}"
                )
            value = row[1].strip()
            if not (value.startswith("[") and value.endswith("]")):
                raise ValueError(f"{filepath}:{expected_index + 1} coordinates must be bracketed")
            coordinate = np.fromstring(value[1:-1], sep=" ", dtype=np.float64)
            if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
                raise ValueError(
                    f"{filepath}:{expected_index + 1} must contain three finite coordinates"
                )
            coords.append(coordinate)
        return np.asarray(coords, dtype=np.float32)
