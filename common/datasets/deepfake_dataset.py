"""
PyTorch dataset for image-level deepfake detection on FF++-style layout.

Features:
- Reads train/val/test splits via manifest files or by scanning the folder tree.
- Identifies Real vs. multiple fake subsets (Deepfakes, Face2Face, FaceSwap,
  NeuralTextures). Label 0 = real, 1 = fake; subtype is kept for analysis.
- Uses dlib frontal face detector for cropping; caches bboxes to avoid repeated
  detection. Cache lives under bbox_cache/ by default.
- Outputs RGB tensors resized to 224x224 (configurable) for dataloader use.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import dlib  # type: ignore

    _HAS_DLIB = True
except ImportError:  # pragma: no cover - optional dependency
    dlib = None
    _HAS_DLIB = False


MANIP_ALIASES: Dict[str, str] = {
    "real": "Real",
    "real_youtube": "Real_youtube",
    "real_youtibe": "Real_youtube",  # typo present in split files
    "deepfakes": "Deepfakes",
    "face2face": "Face2Face",
    "faceswap": "FaceSwap",
    "neuraltextures": "NeuralTextures",
}

# Label 0 = real, 1 = fake
MANIP_LABELS: Dict[str, int] = {
    "Real": 0,
    "Real_youtube": 0,
    "Deepfakes": 1,
    "Face2Face": 1,
    "FaceSwap": 1,
    "NeuralTextures": 1,
}


def _normalize_manip(name: str) -> str:
    key = name.strip()
    key_lower = key.lower()
    return MANIP_ALIASES.get(key, MANIP_ALIASES.get(key_lower, key))


@dataclass
class SampleInfo:
    img_path: Path
    label: int
    manip: str
    video_id: str


class FaceBoxCache:
    """Stores per-frame face boxes to skip repeated dlib calls."""

    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, rel_img: Path) -> Path:
        # Keep folder structure; store one JSON per frame.
        return self.cache_root / rel_img.with_suffix(".bbox.json")

    def load(self, rel_img: Path) -> Optional[Tuple[int, int, int, int]]:
        path = self._cache_path(rel_img)
        if not path.exists():
            return None
        try:
            with path.open("r") as f:
                data = json.load(f)
            return int(data["x0"]), int(data["y0"]), int(data["x1"]), int(data["y1"])
        except Exception:
            return None

    def save(self, rel_img: Path, bbox: Tuple[int, int, int, int]) -> None:
        path = self._cache_path(rel_img)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"x0": int(bbox[0]), "y0": int(bbox[1]), "x1": int(bbox[2]), "y1": int(bbox[3])}
        with path.open("w") as f:
            json.dump(payload, f)


class DeepfakeFaceDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        list_file: Optional[Path] = None,
        frames_per_video: Optional[int] = None,
        frame_sample: str = "uniform",
        bbox_cache_dir: Optional[Path] = None,
        allowed_manips: Optional[Sequence[str]] = None,
        crop_size: int = 224,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_root: Root of dataset, e.g., /ssd2/deepfake/c23.
            split: One of train/val/test.
            list_file: Optional text manifest "<rel_path> <label>" per line.
                       rel_path should point to a video folder under data_root.
            frames_per_video: Limit frames per video. None = all frames.
            frame_sample: "uniform" or "random" sampling when limiting frames.
            bbox_cache_dir: Where to save bbox json files. Defaults to
                            <data_root>/bbox_cache.
            allowed_manips: Optional list of manipulation names to keep
                            (e.g., ["Real", "Deepfakes"]); names are normalized.
            crop_size: Final square size for the face crop.
            transform: Optional transform applied on tensor output.
        """
        # Use absolute root to avoid relative/absolute mismatches when caching bboxes or resolving frames.
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.list_file = Path(list_file) if list_file else None
        self.frames_per_video = frames_per_video
        self.frame_sample = frame_sample
        self.crop_size = crop_size
        self.transform = transform
        self.allowed_manips = set(_normalize_manip(m) for m in allowed_manips) if allowed_manips else None

        cache_dir = bbox_cache_dir or (self.data_root / "bbox_cache")
        self.bbox_cache = FaceBoxCache(cache_dir)

        if not _HAS_DLIB:
            raise ImportError("dlib not available; required for face detection.")
        self.detector = dlib.get_frontal_face_detector()

        self.samples: List[SampleInfo] = self._build_index()

    def _build_index(self) -> List[SampleInfo]:
        if self.list_file:
            videos = self._read_manifest(self.list_file)
        else:
            videos = self._scan_split_dir()

        samples: List[SampleInfo] = []
        for rel_video, label in videos:
            video_dir = self.data_root / rel_video
            if not video_dir.exists():
                raise FileNotFoundError(f"Video folder not found: {video_dir}")
            manip = _normalize_manip(video_dir.parent.name)
            if self.allowed_manips and manip not in self.allowed_manips:
                continue
            video_id = video_dir.name
            frames = self._list_frames(video_dir)
            frames = self._maybe_sample_frames(frames)
            for frame_path in frames:
                rel_frame = frame_path.relative_to(self.data_root)
                samples.append(SampleInfo(img_path=rel_frame, label=label, manip=manip, video_id=video_id))
        return samples

    def _read_manifest(self, path: Path) -> List[Tuple[Path, int]]:
        videos: List[Tuple[Path, int]] = []
        with path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                rel = Path(parts[0])
                lbl = int(parts[1]) if len(parts) > 1 else self._infer_label_from_path(rel)
                # Ensure path contains split prefix.
                if not rel.parts or rel.parts[0] != self.split:
                    rel = Path(self.split) / rel
                # Normalize manipulation folder name to match on-disk layout.
                rel_parts = list(rel.parts)
                if len(rel_parts) >= 2:
                    rel_parts[1] = _normalize_manip(rel_parts[1])
                    rel = Path(*rel_parts)
                videos.append((rel, lbl))
        return videos

    def _scan_split_dir(self) -> List[Tuple[Path, int]]:
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        videos: List[Tuple[Path, int]] = []
        for manip_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            manip = _normalize_manip(manip_dir.name)
            label = MANIP_LABELS.get(manip, 1)
            for vid_dir in sorted(p for p in manip_dir.iterdir() if p.is_dir()):
                rel = Path(self.split) / manip_dir.name / vid_dir.name
                videos.append((rel, label))
        return videos

    def _infer_label_from_path(self, rel: Path) -> int:
        if len(rel.parts) >= 2:
            manip = _normalize_manip(rel.parts[-2])
            return MANIP_LABELS.get(manip, 1)
        return 1

    def _list_frames(self, video_dir: Path) -> List[Path]:
        frames = sorted(video_dir.glob("*.png"))
        if not frames:
            frames = sorted(video_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No frames found in {video_dir}")
        return frames

    def _maybe_sample_frames(self, frames: Sequence[Path]) -> List[Path]:
        if self.frames_per_video is None or len(frames) <= self.frames_per_video:
            return list(frames)
        if self.frame_sample == "random":
            return random.sample(list(frames), self.frames_per_video)
        idxs = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        return [frames[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.samples)

    def _detect_face(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        dets = self.detector(gray, 1)
        if not dets:
            h, w = gray.shape
            side = int(min(h, w) * 0.9)
            cx, cy = w // 2, h // 2
            half = side // 2
            return cx - half, cy - half, cx + half, cy + half
        rect = max(dets, key=lambda r: r.width() * r.height())
        return rect.left(), rect.top(), rect.right(), rect.bottom()

    def _get_face_bbox(self, img_path: Path, gray: np.ndarray) -> Tuple[int, int, int, int]:
        rel = img_path.relative_to(self.data_root)
        cached = self.bbox_cache.load(rel)
        if cached is not None:
            return cached
        bbox = self._detect_face(gray)
        self.bbox_cache.save(rel, bbox)
        return bbox

    def _crop_and_resize(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        h, w, _ = img.shape
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            # Fallback to center crop if detector failed badly.
            side = min(h, w)
            cx, cy = w // 2, h // 2
            half = side // 2
            patch = img[cy - half : cy + half, cx - half : cx + half]
        patch = cv2.resize(patch, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
        return patch

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img_bgr = cv2.imread(str(self.data_root / sample.img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {sample.img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bbox = self._get_face_bbox(self.data_root / sample.img_path, gray)
        face = self._crop_and_resize(img_rgb, bbox)
        tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
        if self.transform:
            tensor = self.transform(tensor)
        return {
            "image": tensor,
            "label": sample.label,
            "manip": sample.manip,
            "video_id": sample.video_id,
            "path": str(sample.img_path),
            "bbox": bbox,
        }


if __name__ == "__main__":
    # Minimal smoke test: sample a few images from train split.
    root = Path("/ssd2/deepfake/c23")
    ds = DeepfakeFaceDataset(data_root=root, split="train", frames_per_video=2, frame_sample="uniform")
    print(f"Found {len(ds)} frames.")
    first = ds[0]
    print("Sample keys:", first.keys())
    print("Image shape:", tuple(first["image"].shape), "Label:", first["label"], "Manip:", first["manip"])
