"""
Multi-modal dataset that aligns RGB / Noise / rPPG inputs using:
- Manifest (JSONL) of videos: split, method, video_id, frames_root, label.
- rPPG cache + meta (produced by rppg/dataset_with_meta.py).
- SamplePlan (precomputed or generated on-the-fly from rPPG meta window).
- FaceTrackCache for unified face bboxes.

This is additive and does not modify existing single-branch datasets.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common.mm.face_track_cache import FaceTrackCache, FaceTrackConfig
from common.mm.sample_plan import make_sample_plan
from common.preprocess.preprocess_pipline_v2 import (  # type: ignore
    build_multiscale_residual_stack,
    select_background_patch,
    to_gray,
    resize_to,
)
from common.datasets.deepfake_dataset import _normalize_manip  # type: ignore


def _video_uid(split: str, method: str, video_id: str) -> str:
    return hashlib.sha1(f"{split}|{method}|{video_id}".encode("utf-8")).hexdigest()


class MultiModalDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        rppg_cache_root: Path = Path("runs/rppg_cache"),
        plan_root: Path = Path("runs/mm_plans"),
        bbox_cache_root: Path = Path("bbox_cache"),
        frames_per_video: int = 8,
        rgb_size: int = 224,
        ms_size: int = 224,
        rppg_size: int = 224,
        strategy: str = "uniform",
        seed: int = 1337,
        allowed_methods: Optional[Sequence[str]] = None,
        allowed_splits: Optional[Sequence[str]] = None,
    ):
        self.allowed_methods = set(_normalize_manip(m) for m in allowed_methods) if allowed_methods else None
        self.allowed_splits = set(s.strip() for s in allowed_splits) if allowed_splits else None
        self.manifest = self._load_manifest(manifest_path)
        self.rppg_cache_root = rppg_cache_root
        self.plan_root = plan_root
        self.frames_per_video = frames_per_video
        self.rgb_size = rgb_size
        self.ms_size = ms_size
        self.rppg_size = rppg_size
        self.strategy = strategy
        self.seed = seed
        self.face_cache = FaceTrackCache(bbox_cache_root, FaceTrackConfig())

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.manifest[idx]
        split = rec["split"]
        method = rec["method"]
        video_id = rec["video_id"]
        label = int(rec.get("label", 0))
        frames_root = Path(rec["frames_root"])
        vid_uid = rec.get("video_uid") or _video_uid(split, method, video_id)

        # rPPG: read cache + meta; if meta missing但有圖，改用全長窗口
        frames_dir = frames_root / split / method / video_id
        rppg_img, rppg_meta = self._load_rppg_cache(split, method, video_id)
        if rppg_img is None:
            raise FileNotFoundError(f"rPPG spectrogram missing for {split}/{method}/{video_id}")
        # 對齊原 rPPG 前處理：等比縮放 + pad/裁切到目標尺寸
        if rppg_img.shape[0] != self.rppg_size or rppg_img.shape[1] != self.rppg_size:
            rppg_img = self._resize_spectrogram_keep_ar(rppg_img, (self.rppg_size, self.rppg_size))
        frames = self._list_frames(frames_dir)
        if not frames:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        if not rppg_meta:
            rppg_meta = {
                "window_start_frame": 0,
                "window_end_frame": len(frames),
                "frame_list_len": len(frames),
                "frame_index_basis": "list_order",
                "frame_list_source": "no_meta_full",
            }

        # plan: load if exists, else create from meta window; fallback to full length
        plan = self._load_plan(vid_uid)
        if plan is None:
            window_start = int(rppg_meta.get("window_start_frame", 0))
            window_end = int(rppg_meta.get("window_end_frame", 0))
            frame_list_len = int(rppg_meta.get("frame_list_len", len(frames)))
            if window_end <= window_start:
                window_end = window_start + frame_list_len
            if window_end <= window_start:
                raise ValueError(f"Invalid window for {vid_uid}: start={window_start}, end={window_end}")
            plan = make_sample_plan(
                video_uid=vid_uid,
                window_start=window_start,
                window_end=window_end,
                full_len=frame_list_len,
                T=self.frames_per_video,
                strategy=self.strategy,
                seed=self.seed,
                frame_index_basis=rppg_meta.get("frame_index_basis", "list_order"),
            )

        # map plan indices to paths
        chosen_paths: List[Path] = []
        for fi in plan.frame_indices:
            if 0 <= fi < len(frames):
                chosen_paths.append(frames[fi])
        if not chosen_paths:
            raise ValueError(f"No valid frames after plan for {vid_uid} (plan={plan.frame_indices}, total_frames={len(frames)})")

        # ensure bboxes
        bbox_map = self.face_cache.ensure(vid_uid, chosen_paths)

        # load RGB stack and Noise stacks
        rgb_stack = []
        face_ms_list = []
        bg_ms_list = []
        for p in chosen_paths:
            bbox = bbox_map.get(p.as_posix())
            img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            if bbox is None:
                h, w, _ = img_bgr.shape
                side = int(min(h, w) * 0.9)
                cx, cy = w // 2, h // 2
                half = side // 2
                bbox = (cx - half, cy - half, cx + half, cy + half)
            rgb_crop = self._crop_and_resize_rgb(img_bgr, bbox, self.rgb_size)
            rgb_stack.append(torch.from_numpy(rgb_crop).permute(2, 0, 1).float() / 255.0)

            gray = to_gray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            face_patch, bg_patch = self._face_bg_patches(gray, bbox)
            face_ms = build_multiscale_residual_stack(face_patch)
            bg_ms = build_multiscale_residual_stack(bg_patch)
            face_ms_list.append(torch.from_numpy(face_ms).float())
            bg_ms_list.append(torch.from_numpy(bg_ms).float())

        rgb_out = torch.stack(rgb_stack, dim=0) if rgb_stack else torch.empty(0)
        face_ms_out = torch.stack(face_ms_list, dim=0) if face_ms_list else torch.empty(0)
        bg_ms_out = torch.stack(bg_ms_list, dim=0) if bg_ms_list else torch.empty(0)

        out = {
            "video_uid": vid_uid,
            "label": torch.tensor(float(label)),
            "rgb": rgb_out,
            "noise_face_ms": face_ms_out,
            "noise_bg_ms": bg_ms_out,
            "rppg": torch.from_numpy(rppg_img).unsqueeze(0).float() / 255.0 if rppg_img is not None else torch.empty(0),
            "meta": {
                "plan": plan.to_dict() if hasattr(plan, "to_dict") else plan,
                "rppg_meta": rppg_meta,
                "frame_indices": plan.frame_indices,
            },
        }
        return out

    # ---------- helpers ----------
    def _load_manifest(self, path: Path) -> List[Dict]:
        items: List[Dict] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if self.allowed_splits and rec.get("split") not in self.allowed_splits:
                    continue
                if self.allowed_methods:
                    method_norm = _normalize_manip(rec.get("method", ""))
                    if method_norm not in self.allowed_methods:
                        continue
                items.append(rec)
        return items

    def _load_plan(self, video_uid: str):
        path = self.plan_root / f"{video_uid}.json"
        if not path.exists():
            return None
        with path.open("r") as f:
            data = json.load(f)
        return type("PlanObj", (), data)  # lightweight

    def _list_frames(self, frames_dir: Path) -> List[Path]:
        list_file = frames_dir / "list.txt"
        if list_file.exists():
            ordered: List[Path] = []
            with list_file.open("r") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        p = frames_dir / name
                        if p.exists():
                            ordered.append(p)
            if ordered:
                return ordered
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            frames = sorted(frames_dir.glob(pattern))
            if frames:
                return frames
        return []

    def _load_rppg_cache(self, split: str, method: str, video_id: str) -> Tuple[Optional[np.ndarray], Dict]:
        meta = {}
        root = self.rppg_cache_root / split / method / video_id
        meta_files = sorted(root.glob("*.meta.json"))
        if meta_files:
            with meta_files[-1].open("r") as f:
                meta = json.load(f)
        img = None
        img_files = sorted(root.glob("*.png"))
        if img_files:
            img = cv2.imread(str(img_files[-1]), cv2.IMREAD_GRAYSCALE)
        return img, meta

    def _resize_spectrogram_keep_ar(self, img: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
        tgt_h, tgt_w = target
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((tgt_h, tgt_w), dtype=np.uint8)
        # 確保 resize 後尺寸不超過目標；保持長寬比
        scale = min(tgt_h / float(h), tgt_w / float(w))
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
        y0 = max(0, (tgt_h - new_h) // 2)
        x0 = max(0, (tgt_w - new_w) // 2)
        y1 = min(tgt_h, y0 + new_h)
        x1 = min(tgt_w, x0 + new_w)
        canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
        return canvas

    def _write_placeholder_meta(
        self, frames_dir: Path, split: str, method: str, video_id: str, rppg_img: np.ndarray
    ) -> Dict:
        frames = self._list_frames(frames_dir)
        if not frames:
            raise FileNotFoundError(f"Cannot build meta; no frames in {frames_dir}")
        frame_list_rel = [p.relative_to(frames_dir.parent.parent.parent).as_posix() for p in frames]
        digest = hashlib.sha1("\n".join(frame_list_rel).encode("utf-8")).hexdigest()
        window_len = len(frames)
        meta = {
            "meta_schema_version": 1,
            "video_uid": _video_uid(split, method, video_id),
            "split": split,
            "method": method,
            "video_id": video_id,
            "fps_used": 30,
            "bandpass_low": 0.7,
            "bandpass_high": 4.0,
            "chrom_enabled": True,
            "detrend_enabled": True,
            "upsample_enabled": True,
            "spec_size": list(rppg_img.shape),
            "window_policy": "full",
            "window_start_frame": 0,
            "window_end_frame": window_len,
            "window_len": window_len,
            "frame_index_basis": "list_order",
            "frame_list_source": "auto_glob",
            "frame_list_len": window_len,
            "path_canonicalization": "relative_to_frames_root_forward_slash",
            "frame_list_digest": digest,
        }
        params_blob = json.dumps(
            {
                "fps_used": meta["fps_used"],
                "bandpass": [meta["bandpass_low"], meta["bandpass_high"]],
                "spec_size": meta["spec_size"],
                "window_policy": meta["window_policy"],
                "window_len": meta["window_len"],
                "chrom": meta["chrom_enabled"],
                "detrend": meta["detrend_enabled"],
                "upsample": meta["upsample_enabled"],
            },
            sort_keys=True,
        )
        meta["params_hash"] = hashlib.sha1(params_blob.encode("utf-8")).hexdigest()
        meta_path = self.rppg_cache_root / split / method / video_id / "auto.meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)
        return meta

    def _crop_and_resize_rgb(self, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int], size: int) -> np.ndarray:
        h, w, _ = img_bgr.shape
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        patch = img_bgr[y0:y1, x0:x1]
        if patch.size == 0:
            side = min(h, w)
            cx, cy = w // 2, h // 2
            half = side // 2
            patch = img_bgr[cy - half : cy + half, cx - half : cx + half]
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)

    def _face_bg_patches(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = gray.shape
        x0, y0, x1, y1 = bbox
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(x0 + 1, min(w, x1))
        y1 = max(y0 + 1, min(h, y1))
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        half = side // 2
        fx0, fx1 = max(0, cx - half), min(w, cx + half)
        fy0, fy1 = max(0, cy - half), min(h, cy + half)
        face = gray[fy0:fy1, fx0:fx1]
        if face.size == 0:
            face = gray
        bg = select_background_patch(gray, (cx, cy), side)
        face = resize_to(face, self.ms_size).astype(np.float32)
        bg = resize_to(bg, self.ms_size).astype(np.float32)
        return face / 255.0, bg / 255.0
