"""
Shared face bbox cache wrapper.

Purpose:
- Single source of truth for face boxes across RGB/Noise branches.
- Batch ensure() to fill missing boxes via dlib (or custom detector).
- Versioned metadata to catch behavior changes.

Usage:
    cache = FaceTrackCache(cache_root=Path("/path/to/bbox_cache"), version="dlib_v1_square")
    boxes = cache.get_many(video_uid, frame_paths)  # returns {frame_path: bbox or None}
    cache.ensure(video_uid, frame_paths)            # will run detector if missing
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    import dlib  # type: ignore

    _HAS_DLIB = True
except ImportError:  # pragma: no cover
    dlib = None
    _HAS_DLIB = False


BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class FaceTrackConfig:
    version: str = "dlib_v1_square"
    square: bool = True
    expand_ratio: float = 1.0  # 1.0 = no expansion
    detector: str = "dlib_frontal"


class FaceTrackCache:
    def __init__(self, cache_root: Path, config: Optional[FaceTrackConfig] = None):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.config = config or FaceTrackConfig()
        if self.config.detector == "dlib_frontal" and not _HAS_DLIB:
            raise ImportError("dlib not available for FaceTrackCache.")
        self.detector = dlib.get_frontal_face_detector() if self.config.detector == "dlib_frontal" else None

    # ---------- public API ----------
    def get_bbox(self, video_uid: str, frame_key: str) -> Optional[BBox]:
        rec = self._load_record(video_uid, frame_key)
        return rec.get("bbox") if rec else None

    def get_many(self, video_uid: str, frame_keys: Iterable[str]) -> Dict[str, Optional[BBox]]:
        out: Dict[str, Optional[BBox]] = {}
        for k in frame_keys:
            rec = self._load_record(video_uid, k)
            out[k] = rec.get("bbox") if rec else None
        return out

    def ensure(self, video_uid: str, frame_paths: List[Path]) -> Dict[str, Optional[BBox]]:
        """
        Ensure bboxes exist for given frame paths. Returns mapping frame_key -> bbox/None.
        frame_key uses posix relative path (portable).
        """
        result: Dict[str, Optional[BBox]] = {}
        for p in frame_paths:
            key = self._frame_key(p)
            rec = self._load_record(video_uid, key)
            bbox = self._normalize_bbox(rec.get("bbox") if rec else None)
            if bbox is None:
                bbox = self._load_sidecar_bbox(p)
                if bbox is not None:
                    self._save_record(video_uid, key, bbox)
            if bbox is None:
                bbox = self._detect(p)
            if bbox is not None:
                self._save_record(video_uid, key, bbox)
            result[key] = bbox
        return result

    # ---------- internal helpers ----------
    def _detect(self, img_path: Path) -> Optional[BBox]:
        if self.detector is None:
            return None
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        dets = self.detector(img, 1)
        if not dets:
            # fallback center crop square
            h, w = img.shape
            side = int(min(h, w) * 0.9)
            cx, cy = w // 2, h // 2
            half = side // 2
            return (cx - half, cy - half, cx + half, cy + half)
        rect = max(dets, key=lambda r: r.width() * r.height())
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        # expand/square
        return self._adjust_bbox((x0, y0, x1, y1), img.shape[:2])

    def _adjust_bbox(self, bbox: BBox, shape_hw: Tuple[int, int]) -> BBox:
        x0, y0, x1, y1 = bbox
        h, w = shape_hw
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        bw, bh = (x1 - x0), (y1 - y0)
        side = max(bw, bh) * self.config.expand_ratio if self.config.square else None
        if side is None:
            x0n, x1n = x0, x1
            y0n, y1n = y0, y1
        else:
            half = side / 2.0
            x0n, x1n = cx - half, cx + half
            y0n, y1n = cy - half, cy + half
        x0n = int(max(0, min(w - 1, x0n)))
        y0n = int(max(0, min(h - 1, y0n)))
        x1n = int(max(x0n + 1, min(w, x1n)))
        y1n = int(max(y0n + 1, min(h, y1n)))
        return (x0n, y0n, x1n, y1n)

    def _rec_path(self, video_uid: str, frame_key: str) -> Path:
        return self.cache_root / video_uid / (frame_key + ".bbox.json")

    def _load_record(self, video_uid: str, frame_key: str) -> Optional[Dict]:
        path = self._rec_path(video_uid, frame_key)
        if not path.exists():
            return None
        try:
            with path.open("r") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_record(self, video_uid: str, frame_key: str, bbox: BBox) -> None:
        path = self._rec_path(video_uid, frame_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bbox": [int(x) for x in bbox],
            "version": self.config.version,
            "detector": self.config.detector,
            "expand_ratio": self.config.expand_ratio,
            "square": self.config.square,
        }
        with path.open("w") as f:
            json.dump(payload, f)

    @staticmethod
    def _frame_key(p: Path) -> str:
        # portable, relative string with forward slashes
        return p.as_posix()

    @staticmethod
    def _normalize_bbox(bbox: Optional[object]) -> Optional[BBox]:
        if bbox is None:
            return None
        if isinstance(bbox, dict):
            bbox = bbox.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                return tuple(int(x) for x in bbox)  # type: ignore[return-value]
            except Exception:
                return None
        return None

    def _load_sidecar_bbox(self, frame_path: Path) -> Optional[BBox]:
        """
        Read bbox saved next to the frame (e.g., 000.png.bbox.json) if available.
        """
        sidecar = Path(str(frame_path) + ".bbox.json")
        if not sidecar.exists():
            return None
        try:
            with sidecar.open("r") as f:
                data = json.load(f)
            return self._normalize_bbox(data)
        except Exception:
            return None
