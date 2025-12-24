"""
Video-level RGB face dataset (v2) that returns T-frame bags per video.

Each item packs:
    rgb_face: (T, 3, crop_size, crop_size) float [0,1] face crops
    label:    0 real / 1 fake
    manip:    subtype name (e.g., Deepfakes)
    video_id: video folder name
    frame_idx: (T,) long tensor parsed from filenames (or -1)
    paths:   list[str] relative frame paths used for this sample

Indexing:
    - Scans split folder or manifest to build a video-level list:
      {"video_id": "...", "label": 0/1, "manip": "...", "frames": [rel paths]}
    - __getitem__ samples frames_per_video frames from the same video
      (uniform / random / fixed-deterministic) and returns stacked tensors.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.datasets.deepfake_dataset import FaceBoxCache, MANIP_LABELS, _normalize_manip  # type: ignore
from common.mm.face_track_cache import FaceTrackCache  # optional shared cache

try:
    import dlib  # type: ignore

    _HAS_DLIB = True
except ImportError:  # pragma: no cover
    dlib = None
    _HAS_DLIB = False


class RgbVideoBagDatasetV2(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        index_list: Optional[List[Dict[str, Any]]] = None,
        list_file: Optional[Path] = None,
        frames_per_video: int = 8,
        frame_sample: str = "uniform",
        bbox_cache_dir: Optional[Path] = None,
        allowed_manips: Optional[Sequence[str]] = None,
        crop_size: int = 224,
        transform=None,
        sample_plan: Optional[Dict[str, Any]] = None,
        face_cache: Optional[FaceTrackCache] = None,
    ):
        """
        Args mirror deepfake_dataset but operate on video-level bags.
        index_list: pre-built video index (list of dicts). If None, build by scanning / manifest.
        frames_per_video: T frames sampled per video when fetching an item.
        frame_sample: "uniform" | "random" | "fixed" (deterministic per video).
        """
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.list_file = Path(list_file) if list_file else None
        self.frames_per_video = frames_per_video
        self.frame_sample = frame_sample
        self.crop_size = crop_size
        self.transform = transform
        self.allowed_manips = set(_normalize_manip(m) for m in allowed_manips) if allowed_manips else None
        self.sample_plan = sample_plan  # optional, dict with frame_indices
        self.face_cache = face_cache

        if not _HAS_DLIB:
            raise ImportError("dlib not available; required for face detection.")
        self.detector = dlib.get_frontal_face_detector()

        cache_dir = bbox_cache_dir or (self.data_root / "bbox_cache")
        self.bbox_cache = FaceBoxCache(cache_dir)

        self.videos = self._normalize_index(index_list) if index_list is not None else self._build_index()

    # ---------- index helpers ----------
    def _normalize_index(self, index_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for meta in index_list:
            manip = _normalize_manip(str(meta.get("manip", "")))
            if self.allowed_manips and manip not in self.allowed_manips:
                continue
            frames = [Path(f) for f in meta.get("frames", [])]
            normalized.append(
                {
                    "video_id": meta.get("video_id", Path(frames[0]).parent.name if frames else ""),
                    "label": int(meta.get("label", self._infer_label_from_path(frames[0] if frames else Path()))),
                    "manip": manip if manip else "Unknown",
                    "frames": frames,
                }
            )
        return normalized

    def _build_index(self) -> List[Dict[str, Any]]:
        videos = self._read_manifest(self.list_file) if self.list_file else self._scan_split_dir()
        index: List[Dict[str, Any]] = []
        for rel_video, label, manip in videos:
            video_dir = self.data_root / rel_video
            if not video_dir.exists():
                raise FileNotFoundError(f"Video folder not found: {video_dir}")
            frames = self._list_frames(video_dir)
            if not frames:
                continue
            rel_frames = [p.relative_to(self.data_root) for p in frames]
            index.append(
                {"video_id": video_dir.name, "label": label, "manip": manip, "frames": rel_frames}
            )
        return index

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
        frames = sorted(video_dir.glob("*.png"))
        if not frames:
            frames = sorted(video_dir.glob("*.jpg"))
        if not frames:
            frames = sorted(video_dir.glob("*.jpeg"))
        return frames

    def __len__(self) -> int:
        return len(self.videos)

    # ---------- detection & caching ----------
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
            side = min(h, w)
            cx, cy = w // 2, h // 2
            half = side // 2
            patch = img[cy - half : cy + half, cx - half : cx + half]
        return cv2.resize(patch, (self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)

    # ---------- frame sampling ----------
    def _sample_frames(self, frames: Sequence[Path], rel_video: str) -> List[Path]:
        if self.sample_plan and self.sample_plan.get("frame_indices"):
            idxs = [int(i) for i in self.sample_plan["frame_indices"]]
            return [frames[i] for i in idxs if 0 <= i < len(frames)]
        if self.frames_per_video is None or len(frames) <= self.frames_per_video:
            return list(frames)
        if self.frame_sample == "random":
            return random.sample(list(frames), self.frames_per_video)
        if self.frame_sample == "fixed":
            seed = int(hashlib.md5(rel_video.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            idxs = rng.choice(len(frames), size=self.frames_per_video, replace=False)
            return [frames[i] for i in sorted(idxs)]
        idxs = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        return [frames[i] for i in idxs]

    # ---------- main ----------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.videos[idx]
        frame_paths: List[Path] = meta["frames"]
        chosen_paths = self._sample_frames(frame_paths, rel_video=meta.get("video_id", ""))

        rgb_list, frame_idx_list, bboxes = [], [], []
        for rel_path in chosen_paths:
            img_path = self.data_root / rel_path
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            bbox = None
            if self.face_cache is not None:
                rec = self.face_cache.get_bbox(self._video_uid(rel_path), rel_path.as_posix())
                bbox = tuple(rec) if rec is not None else None
            if bbox is None:
                bbox = self._get_face_bbox(img_path, gray)
            face = self._crop_and_resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bbox)
            tensor = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
            if self.transform:
                tensor = self.transform(tensor)
            rgb_list.append(tensor)
            frame_idx_list.append(parse_frame_idx(rel_path.name))
            bboxes.append(bbox)

        rgb_stack = torch.stack(rgb_list, dim=0)  # (T,3,H,W)

        return {
            "rgb_face": rgb_stack,
            "label": torch.tensor(meta["label"], dtype=torch.float32),
            "manip": meta["manip"],
            "video_id": meta["video_id"],
            "frame_idx": torch.tensor(frame_idx_list, dtype=torch.long),
            "paths": [str(p) for p in chosen_paths],
            "bbox": bboxes,
        }

    def _video_uid(self, rel_path: Path) -> str:
        parts = rel_path.parts
        # Expect split/method/video/frame
        if len(parts) >= 3:
            return hashlib.sha1(f"{parts[0]}|{parts[1]}|{parts[2]}".encode("utf-8")).hexdigest()
        return hashlib.sha1(str(rel_path).encode("utf-8")).hexdigest()


def parse_frame_idx(name: str) -> int:
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return -1
