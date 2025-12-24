"""
Dataset that combines RGB face crops with NoiseDF-lite multi-scale residuals.

Outputs (dict; keys are optional based on flags):
    rgb_face: (3, crop_size, crop_size) float [0,1] RGB face crop
    face_ms: (3, ms_size, ms_size) float multi-scale residual stack (face)
    bg_ms:   (3, ms_size, ms_size) float multi-scale residual stack (background)
    face_residual / bg_residual: (ms_size, ms_size) single-scale residuals (debug)
    bbox: (x0,y0,x1,y1) face bbox
    label: 0 real / 1 fake
    manip: subtype name
    video_id: video folder name
    frame_idx: int parsed from filename if possible else -1
    path: relative path to the frame

Supports:
- Scan split folders or read manifest (same style as deepfake_dataset.py).
- dlib face detection with bbox caching under <data_root>/bbox_cache (or custom).
- Optional precompute_dir: if provided, face_ms/bg_ms will be cached as .npy and
  reused to avoid recomputing filters every epoch.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    import dlib  # type: ignore

    _HAS_DLIB = True
except ImportError:  # pragma: no cover
    dlib = None
    _HAS_DLIB = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.datasets.deepfake_dataset import FaceBoxCache, MANIP_LABELS, _normalize_manip  # type: ignore
from common.mm.face_track_cache import FaceTrackCache  # optional shared cache

# Reuse functions from preprocess_pipline to stay consistent.
from common.preprocess.preprocess_pipline import (  # type: ignore
    build_multiscale_residual_stack,
    compute_residual,
    resize_to,
    select_background_patch,
    to_gray,
)


class NoiseFaceDataset:
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
        ms_size: int = 128,
        include_rgb: bool = False,
        include_ms_face: bool = True,
        include_ms_bg: bool = True,
        include_residual: bool = False,
        precompute_dir: Optional[Path] = None,
        sample_plan: Optional[Dict] = None,
        face_cache: Optional[FaceTrackCache] = None,
    ):
        """
        Args mirror deepfake_dataset with extra toggles for noise outputs.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.list_file = Path(list_file) if list_file else None
        self.frames_per_video = frames_per_video
        self.frame_sample = frame_sample
        self.crop_size = crop_size
        self.ms_size = ms_size
        self.include_rgb = include_rgb
        self.include_ms_face = include_ms_face
        self.include_ms_bg = include_ms_bg
        self.include_residual = include_residual
        self.allowed_manips = set(_normalize_manip(m) for m in allowed_manips) if allowed_manips else None
        self.sample_plan = sample_plan
        self.face_cache = face_cache

        if not _HAS_DLIB:
            raise ImportError("dlib not available; required for face detection.")
        self.detector = dlib.get_frontal_face_detector()

        cache_dir = bbox_cache_dir or (self.data_root / "bbox_cache")
        self.bbox_cache = FaceBoxCache(cache_dir)

        self.precompute_dir = Path(precompute_dir) if precompute_dir else None
        if self.precompute_dir:
            (self.precompute_dir).mkdir(parents=True, exist_ok=True)

        self.samples = self._build_index()

    # ---------- index helpers ----------
    def _build_index(self) -> List[Tuple[Path, int, str]]:
        videos = self._read_manifest(self.list_file) if self.list_file else self._scan_split_dir()
        samples: List[Tuple[Path, int, str]] = []
        for rel_video, label, manip in videos:
            video_dir = self.data_root / rel_video
            if not video_dir.exists():
                raise FileNotFoundError(f"Video folder not found: {video_dir}")
            frames = self._list_frames(video_dir)
            frames = self._maybe_sample_frames(frames)
            for frame_path in frames:
                rel_frame = frame_path.relative_to(self.data_root)
                samples.append((rel_frame, label, manip))
        return samples

    def _read_manifest(self, path: Path) -> List[Tuple[Path, int, str]]:
        videos: List[Tuple[Path, int, str]] = []
        with path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                rel = Path(parts[0])
                lbl = int(parts[1]) if len(parts) > 1 else self._infer_label_from_path(rel)
                if not rel.parts or rel.parts[0] != self.split:
                    rel = Path(self.split) / rel
                rel_parts = list(rel.parts)
                if len(rel_parts) >= 2:
                    rel_parts[1] = _normalize_manip(rel_parts[1])
                    rel = Path(*rel_parts)
                manip = _normalize_manip(rel_parts[1]) if len(rel_parts) >= 2 else "Unknown"
                if self.allowed_manips and manip not in self.allowed_manips:
                    continue
                videos.append((rel, lbl, manip))
        return videos

    def _scan_split_dir(self) -> List[Tuple[Path, int, str]]:
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        videos: List[Tuple[Path, int, str]] = []
        for manip_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            manip = _normalize_manip(manip_dir.name)
            if self.allowed_manips and manip not in self.allowed_manips:
                continue
            label = MANIP_LABELS.get(manip, 1)
            for vid_dir in sorted(p for p in manip_dir.iterdir() if p.is_dir()):
                rel = Path(self.split) / manip_dir.name / vid_dir.name
                videos.append((rel, label, manip))
        return videos

    def _infer_label_from_path(self, rel: Path) -> int:
        if len(rel.parts) >= 2:
            manip = _normalize_manip(rel.parts[-2])
            return MANIP_LABELS.get(manip, 1)
        return 1

    def _list_frames(self, video_dir: Path) -> List[Path]:
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            frames = sorted(video_dir.glob(pattern))
            if frames:
                return frames
        raise FileNotFoundError(f"No frames found in {video_dir}")

    def _maybe_sample_frames(self, frames: Sequence[Path]) -> List[Path]:
        if self.sample_plan and self.sample_plan.get("frame_indices"):
            idxs = [int(i) for i in self.sample_plan["frame_indices"]]
            return [frames[i] for i in idxs if 0 <= i < len(frames)]
        if self.frames_per_video is None or len(frames) <= self.frames_per_video:
            return list(frames)
        if self.frame_sample == "random":
            return random.sample(list(frames), self.frames_per_video)
        idxs = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        return [frames[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.samples)

    # ---------- detection & caching ----------
    def _detect_face(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        if gray.max() <= 1.5:
            gray_u8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        dets = self.detector(gray_u8, 1)
        if not dets:
            h, w = gray_u8.shape
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

    # ---------- precompute helpers ----------
    def _precompute_path(self, rel_img: Path, kind: str) -> Path:
        if self.precompute_dir is None:
            raise ValueError("precompute_dir not set")
        subdir = self.precompute_dir / f"ms{self.ms_size}"
        return subdir / rel_img.with_suffix(f".{kind}.npy")

    def _load_or_compute_ms(self, rel_img: Path, gray_crop: np.ndarray, kind: str) -> np.ndarray:
        if self.precompute_dir:
            path = self._precompute_path(rel_img, kind)
            if path.exists():
                try:
                    cached = np.load(path)
                    if cached.shape == (3, self.ms_size, self.ms_size):
                        return cached
                except Exception:
                    # Corrupted cache; fall through to recompute.
                    pass
            ms = build_multiscale_residual_stack(gray_crop)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, ms)
            return ms
        return build_multiscale_residual_stack(gray_crop)

    # ---------- main ----------
    def __getitem__(self, idx: int) -> Dict:
        rel_path, label, manip = self.samples[idx]
        img_path = self.data_root / rel_path
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Decode once from the original JPEG/MP4 frame; keep gray for noise, RGB only if requested.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = to_gray(img_rgb)
        bbox = None
        if self.face_cache is not None:
            rec = self.face_cache.get_bbox(self._video_uid(rel_path), str(rel_path))
            bbox = tuple(rec) if rec is not None else None
        if bbox is None:
            bbox = self._get_face_bbox(img_path, gray)
        x0, y0, x1, y1 = bbox
        h, w = gray.shape
        raw_side = max(x1 - x0, y1 - y0)
        max_side = int(0.8 * min(h, w))
        side = max(32, min(int(raw_side * 1.3), max_side))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        face_patch = crop_square_safe(gray, (cx, cy), side)
        bg_patch = select_background_patch(gray, (cx, cy), side)

        face_128 = resize_to(face_patch, self.ms_size)
        bg_128 = resize_to(bg_patch, self.ms_size)

        out: Dict = {
            "label": label,
            "manip": manip,
            "video_id": rel_path.parent.name,
            "frame_idx": parse_frame_idx(rel_path.name),
            "path": str(rel_path),
            "bbox": bbox,
        }

        rel_for_cache = rel_path
        if self.include_ms_face:
            ms = self._load_or_compute_ms(rel_for_cache, face_128, "face_ms")
            out["face_ms"] = torch.from_numpy(ms).float()
        if self.include_ms_bg:
            ms = self._load_or_compute_ms(rel_for_cache, bg_128, "bg_ms")
            out["bg_ms"] = torch.from_numpy(ms).float()
        if self.include_residual:
            out["face_residual"] = torch.from_numpy(compute_residual(face_128)).float()
            out["bg_residual"] = torch.from_numpy(compute_residual(bg_128)).float()
        if self.include_rgb:
            face_rgb = crop_and_resize_rgb(img_rgb, bbox, self.crop_size)
            face_rgb = face_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            out["rgb_face"] = torch.from_numpy(face_rgb).float()

        return out

    def _video_uid(self, rel_path: Path) -> str:
        parts = rel_path.parts
        if len(parts) >= 3:
            return hashlib.sha1(f"{parts[0]}|{parts[1]}|{parts[2]}".encode("utf-8")).hexdigest()
        return hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()


# ---------- small utilities ----------

def crop_square_safe(gray: np.ndarray, center: Tuple[int, int], side: int) -> np.ndarray:
    h, w = gray.shape
    cx, cy = center
    half = side // 2
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    return gray[y0:y1, x0:x1]


def crop_and_resize_rgb(img: np.ndarray, bbox: Tuple[int, int, int, int], size: int) -> np.ndarray:
    h, w, _ = img.shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        side = min(h, w)
        cx, cy = w // 2, h // 2
        half = side // 2
        patch = img[cy - half : cy + half, cx - half : cx + half]
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_CUBIC)


def parse_frame_idx(name: str) -> int:
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return -1


if __name__ == "__main__":
    root = Path("/ssd2/deepfake/c23")
    ds = NoiseFaceDataset(
        data_root=root,
        split="train",
        frames_per_video=2,
        include_rgb=True,
        include_ms_face=True,
        include_ms_bg=True,
        include_residual=False,
        precompute_dir=root / "proc_cache",
    )
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(sample.keys())
