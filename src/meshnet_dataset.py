from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset as TorchDataset

from .dataset import Dataset as MeshLandmarkDataset
from .ear_crop import make_crop_target
from .preprocessing import MeshNormalization, compute_mesh_normalization

EAR_NAMES = ("left", "right")
EPSILON = 1e-8


def compute_face_features(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return (F, 15) features: center (3), corners (9), unit normal (3)."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = vertices[faces]  # (F, 3, 3)

    centers = triangles.mean(axis=1)
    corners = triangles.reshape(len(faces), 9)

    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    normals = normals / norms

    return np.concatenate([centers, corners, normals], axis=1).astype(np.float32)


def compute_face_neighbors(faces: np.ndarray) -> np.ndarray:
    """Return (F, 3) neighbor indices via shared edges (self-index if none)."""
    faces = np.asarray(faces, dtype=np.int64)
    edge_to_faces: Dict[Tuple[int, int], list] = {}
    for face_idx, (v1, v2, v3) in enumerate(faces):
        for a, b in ((v1, v2), (v2, v3), (v3, v1)):
            edge = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(edge, []).append(face_idx)

    neighbors = np.empty((len(faces), 3), dtype=np.int64)
    for face_idx, (v1, v2, v3) in enumerate(faces):
        for slot, (a, b) in enumerate(((v1, v2), (v2, v3), (v3, v1))):
            edge = (a, b) if a < b else (b, a)
            neighbor = face_idx
            for candidate in edge_to_faces[edge]:
                if candidate != face_idx:
                    neighbor = candidate
                    break
            neighbors[face_idx, slot] = neighbor
    return neighbors


def simplify_mesh_to_max_faces(
    mesh: trimesh.Trimesh, max_faces: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) with at most max_faces faces.

    Prefers quadric decimation (preserves edge-adjacency structure); falls
    back to random face subsampling if decimation is unavailable.
    """
    if len(mesh.faces) <= max_faces:
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)

    simplified = None
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
    except (TypeError, AttributeError, ImportError, ValueError):
        try:
            simplified = mesh.simplify_quadric_decimation(max_faces)
        except Exception:
            simplified = None
    except Exception:
        simplified = None

    if simplified is not None and len(simplified.faces) > 0:
        vertices = np.asarray(simplified.vertices)
        faces = np.asarray(simplified.faces)
    else:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

    if len(faces) > max_faces:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(faces), size=max_faces, replace=False)
        faces = faces[np.sort(keep)]

    return vertices, faces


class PinnaPrecropMeshDataset(TorchDataset):
    """Return MeshNet face features for one precropped ear and its landmarks."""

    def __init__(
        self,
        mesh_dir: str,
        landmarks_dir: str,
        cropped_dir: str,
        max_faces: int = 1024,
        seed: int = 0,
        subject_ids: Optional[Sequence[str]] = None,
        mirror_right_ear: bool = False,
        augment: bool = False,
        jitter_sigma: float = 0.002,
        jitter_clip: float = 0.01,
        face_cache_dir: Optional[str] = None,
    ):
        self.base_dataset = MeshLandmarkDataset(mesh_dir=mesh_dir, landmarks_dir=landmarks_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.cropped_dir = Path(cropped_dir)
        self.max_faces = int(max_faces)
        self.seed = int(seed)
        self.mirror_right_ear = bool(mirror_right_ear)
        self.augment = bool(augment)
        self.jitter_sigma = float(jitter_sigma)
        self.jitter_clip = float(jitter_clip)
        self.face_cache_dir = Path(face_cache_dir) if face_cache_dir else None
        if self.face_cache_dir is not None:
            self.face_cache_dir.mkdir(parents=True, exist_ok=True)

        if subject_ids is None:
            self.indices = list(range(len(self.base_dataset)))
        else:
            wanted = {subject_id for subject_id in subject_ids}
            id_to_index = {
                self.base_dataset.get_identifier(idx): idx for idx in range(len(self.base_dataset))
            }
            missing = sorted(wanted - set(id_to_index))
            if missing:
                raise ValueError(f"Unknown subject ids: {missing}")
            self.indices = [id_to_index[subject_id] for subject_id in subject_ids]

        self.samples = [
            (base_idx, ear)
            for base_idx in self.indices
            for ear in EAR_NAMES
        ]
        self._crop_index = self._build_crop_index()
        self._validate_cropped_files()
        self._memory_cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}

    def _build_crop_index(self) -> Dict[str, Path]:
        """Map '{subject}_{ear}_mesh.ply' filename -> full path.

        Searches cropped_dir recursively so both a flat layout and the
        box-regressor export layout (crops/<split>/{subject}_{ear}_mesh.ply)
        are supported. If the same filename appears in multiple subfolders,
        the shallowest path wins (deterministic by path depth then name).
        """
        if not self.cropped_dir.exists():
            raise FileNotFoundError(f"cropped-dir does not exist: {self.cropped_dir}")
        candidates = sorted(
            self.cropped_dir.rglob("*_mesh.ply"),
            key=lambda p: (len(p.relative_to(self.cropped_dir).parts), str(p)),
        )
        index: Dict[str, Path] = {}
        for path in candidates:
            index.setdefault(path.name, path)
        return index

    def _crop_path(self, subject_id: str, ear: str) -> Optional[Path]:
        return self._crop_index.get(f"{subject_id}_{ear}_mesh.ply")

    def _validate_cropped_files(self) -> None:
        missing = []
        for base_idx, ear in self.samples:
            subject_id = self.base_dataset.get_identifier(base_idx)
            if self._crop_path(subject_id, ear) is None:
                missing.append(f"{subject_id}_{ear}_mesh.ply")
        if missing:
            preview = "\n".join(missing[:10])
            found = len(self._crop_index)
            sample_found = "\n".join(sorted(self._crop_index)[:5])
            raise FileNotFoundError(
                f"{len(missing)} cropped ear mesh file(s) not found under {self.cropped_dir} "
                f"(searched recursively; found {found} '*_mesh.ply' file(s)).\n"
                f"Missing (by expected filename):\n{preview}\n"
                f"Examples of files that WERE found:\n{sample_found or '(none)'}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, subject_id: str, ear: str) -> Optional[Path]:
        if self.face_cache_dir is None:
            return None
        return self.face_cache_dir / f"{subject_id}_{ear}_faces{self.max_faces}.npz"

    def _get_processed(
        self, base_idx: int, ear: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return (face_features (F,15), neighbors (F,3), centroid (3,), scale)."""
        subject_id = self.base_dataset.get_identifier(base_idx)
        key = (subject_id, ear)
        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_path = self._cache_path(subject_id, ear)
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path)
            result = (
                data["faces"].astype(np.float32),
                data["neighbors"].astype(np.int64),
                data["centroid"].astype(np.float32),
                float(data["scale"]),
            )
            self._memory_cache[key] = result
            return result

        # Full mesh only used for the global normalization transform.
        full_mesh, _, _ = self.base_dataset[base_idx]
        transform = compute_mesh_normalization(full_mesh)

        crop_path = self._crop_path(subject_id, ear)
        if crop_path is None:
            raise FileNotFoundError(
                f"No cropped mesh for {subject_id} {ear} under {self.cropped_dir}"
            )
        crop_mesh = trimesh.load(crop_path, force="mesh")
        faces = np.asarray(crop_mesh.faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(
                f"Cropped mesh has no triangle faces: {crop_path}. "
                "MeshNet requires a triangle mesh (the *_mesh.ply crops, not *_points.ply)."
            )

        ear_offset = 0 if ear == "left" else 1
        vertices, faces = simplify_mesh_to_max_faces(
            crop_mesh, self.max_faces, seed=self.seed + base_idx * 2 + ear_offset
        )
        vertices = transform.normalize_xyz(vertices.astype(np.float32))
        face_features = compute_face_features(vertices, faces)
        neighbors = compute_face_neighbors(faces)

        result = (face_features, neighbors, transform.centroid.astype(np.float32), transform.scale)
        if cache_path is not None:
            np.savez(
                cache_path,
                faces=face_features,
                neighbors=neighbors,
                centroid=result[2],
                scale=np.float32(result[3]),
            )
        self._memory_cache[key] = result
        return result

    def __getitem__(self, idx: int) -> dict:
        base_idx, ear = self.samples[idx]
        subject_id = self.base_dataset.get_identifier(base_idx)
        face_features, neighbors, centroid, scale = self._get_processed(base_idx, ear)
        face_features = face_features.copy()
        neighbors = neighbors.copy()
        transform = MeshNormalization(centroid=centroid, scale=scale)

        landmarks_path = (
            self.landmarks_dir / f"{subject_id}_{ear}_ear_landmarks.csv"
        )
        landmarks = self.base_dataset._load_landmarks(landmarks_path)
        target = make_crop_target(landmarks, transform)

        # Pad with randomly repeated faces up to max_faces (official MeshNet style).
        num_faces = len(face_features)
        if num_faces < self.max_faces:
            rng = np.random.default_rng(self.seed + base_idx * 2 + (0 if ear == "left" else 1))
            fill = rng.integers(0, num_faces, size=self.max_faces - num_faces)
            face_features = np.concatenate([face_features, face_features[fill]], axis=0)
            neighbors = np.concatenate([neighbors, neighbors[fill]], axis=0)

        mirrored = ear == "right" and self.mirror_right_ear
        if mirrored:
            # Reflect input AND target across the Y plane so the pair stays
            # consistent. Inference must un-mirror right-ear predictions.
            face_features[:, [1, 4, 7, 10, 13]] *= -1.0
            target = target.copy()
            target[:, 1] *= -1.0

        face = torch.from_numpy(face_features).float().permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        if self.augment:
            jitter = np.clip(
                self.jitter_sigma * np.random.randn(3, centers.shape[1]),
                -self.jitter_clip,
                self.jitter_clip,
            ).astype(np.float32)
            centers = centers + torch.from_numpy(jitter)

        return {
            "centers": centers,
            "corners": corners,
            "normals": normals,
            "neighbor_index": torch.from_numpy(neighbors).long(),
            "landmarks": torch.from_numpy(target),
            "centroid": torch.from_numpy(centroid),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "identifier": subject_id,
            "ear": ear,
            "mirrored": mirrored,
        }