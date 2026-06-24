from typing import Iterable, Optional, Tuple
import numpy as np
import trimesh
from trimesh import Trimesh
import csv
from pathlib import Path


DEFAULT_EXCLUDED_SUBJECT_IDS = {"P0027"}


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
        mesh = trimesh.load(mesh_path)

        # --- Load landmarks ---
        left_path = self.landmarks_dir / f"{subject_id}_left_ear_landmarks.csv"
        right_path = self.landmarks_dir/ f"{subject_id}_right_ear_landmarks.csv"
        landmarks_left = self._load_landmarks(left_path)
        landmarks_right = self._load_landmarks(right_path)

        return mesh, landmarks_left, landmarks_right    
    
    def _load_landmarks(self, filepath: Path) -> np.ndarray:
        """
        input CSV with format: index, [x y z]
        output N x 3 array
        """
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            coords = [np.fromstring(coordinate_str.strip('[]'), sep=' ') for _, coordinate_str in reader]
        return np.array(coords)
