"""
Video-level rPPG dataset that mirrors rppg/dataset.py but also writes a
sidecar meta JSON next to each cached spectrogram. Existing pipelines stay
unchanged; use this class when you need rPPG cache with traceable metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import butter, detrend, filtfilt, spectrogram, welch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoRPPGDatasetWithMeta(Dataset):
    """
    Generate one rPPG spectrogram per video and store a meta JSON alongside the
    cached image. Directory layout: frames_root/<split>/<method>/<video_id>/*.png
    """

    def __init__(
        self,
        frames_root: str,
        split: str,
        rgb_size: int = 224,
        rppg_size: int = 224,
        frames_per_video: Optional[int] = None,
        frame_sample: str = "uniform",
        allowed_methods: Optional[Sequence[str]] = None,
        rppg_cache_dir: Optional[str] = None,
        rppg_window_size: Optional[int] = None,
        rppg_window_seconds: Optional[float] = None,
        rppg_fps: int = 30,
        fps_map: Optional[Dict[str, int]] = None,
        spec_clip: Tuple[float, float] = (-8.0, 2.0),
        spec_norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.frames_root = Path(frames_root)
        self.split = split
        self.rgb_size = rgb_size
        self.rppg_size = rppg_size
        self.frames_per_video = frames_per_video
        self.frame_sample = frame_sample
        self.allowed_methods = set(allowed_methods) if allowed_methods else None
        self.rppg_cache_dir = Path(rppg_cache_dir) if rppg_cache_dir else None
        self.rppg_window_size = rppg_window_size
        self.rppg_window_seconds = rppg_window_seconds
        self.rppg_fps = rppg_fps
        self.fps_map = fps_map or {}
        self.spec_clip = spec_clip
        self.spec_norm_stats = spec_norm_stats  # (mu_f, sigma_f) if provided

        self.rppg_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        )

        self.samples = self._build_index()

    # ---------- index helpers ----------
    def _build_index(self) -> List[Tuple[Path, int, str]]:
        split_dir = self.frames_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        samples: List[Tuple[Path, int, str]] = []
        for method_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            method = method_dir.name
            if self.allowed_methods and method not in self.allowed_methods:
                continue
            label = 0 if method.lower().startswith("real") else 1
            for vid_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                samples.append((vid_dir, label, method))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _list_frames(self, video_dir: Path) -> Tuple[List[Path], str]:
        list_file = video_dir / "list.txt"
        if list_file.exists():
            ordered: List[Path] = []
            try:
                with list_file.open("r") as f:
                    for line in f:
                        name = line.strip()
                        if not name:
                            continue
                        p = video_dir / name
                        if p.exists():
                            ordered.append(p)
                if ordered:
                    return ordered, "list.txt"
            except Exception:
                pass

        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            frames = sorted(video_dir.glob(pattern))
            if frames:
                return frames, f"glob:{pattern}"
        return [], "glob:none"

    def _maybe_sample_frames(self, frames: Sequence[Path], rel_video: str) -> List[Path]:
        if self.frames_per_video is None or len(frames) <= self.frames_per_video:
            return list(frames)
        if self.frame_sample == "random":
            idxs = np.random.choice(len(frames), size=self.frames_per_video, replace=False)
            return [frames[i] for i in sorted(idxs)]
        # default uniform/fixed
        idxs = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        return [frames[i] for i in idxs]

    # ---------- main ----------
    def __getitem__(self, idx: int):
        video_dir, label, method = self.samples[idx]
        frames, frame_source = self._list_frames(video_dir)
        if not frames:
            raise FileNotFoundError(f"No frames found in {video_dir}")
        frames = self._maybe_sample_frames(frames, rel_video=str(video_dir.relative_to(self.frames_root)))

        cache_img = self._load_cache(video_dir)
        if cache_img is None:
            cache_img = self._compute_rppg(frames, video_dir.name)
            if cache_img is None:
                raise IndexError("rPPG computation failed; skip sample")
            self._save_cache(video_dir, cache_img, frames, frame_source)

        I_rppg = self.rppg_transform(Image.fromarray(cache_img).convert("L"))
        return I_rppg, torch.tensor(float(label))

    # ---- cache helpers ----
    def _cache_key(self, video_dir: Path) -> str:
        h = hashlib.md5()
        h.update(str(video_dir).encode("utf-8"))
        h.update(f"{self.rppg_window_size}_{self.rppg_fps}".encode("utf-8"))
        return h.hexdigest()

    def _cache_img_path(self, video_dir: Path) -> Path:
        key = self._cache_key(video_dir) + ".png"
        return self._cache_root(video_dir) / key

    def _cache_meta_path(self, video_dir: Path) -> Path:
        key = self._cache_key(video_dir) + ".meta.json"
        return self._cache_root(video_dir) / key

    def _cache_root(self, video_dir: Path) -> Path:
        if not self.rppg_cache_dir:
            return Path()
        return self.rppg_cache_dir / self.split / video_dir.parts[-2] / video_dir.name

    def _load_cache(self, video_dir: Path) -> Optional[np.ndarray]:
        if not self.rppg_cache_dir:
            return None
        cache_path = self._cache_img_path(video_dir)
        if cache_path.exists():
            try:
                img = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
            except Exception:
                return None
        return None

    def _save_cache(self, video_dir: Path, img: np.ndarray, frames: Sequence[Path], frame_source: str):
        if not self.rppg_cache_dir:
            return
        cache_img = self._cache_img_path(video_dir)
        cache_img.parent.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(str(cache_img), img)
        except Exception:
            pass
        # sidecar meta
        meta_path = self._cache_meta_path(video_dir)
        frame_list_rel = [p.relative_to(self.frames_root).as_posix() for p in frames]
        frame_list_digest = hashlib.sha1("\n".join(frame_list_rel).encode("utf-8")).hexdigest()
        fps_used = self.fps_map.get(video_dir.name, self.rppg_fps)
        window_len = len(frames)
        meta = {
            "meta_schema_version": 1,
            "video_uid": hashlib.sha1(f"{self.split}|{video_dir.parts[-2]}|{video_dir.name}".encode("utf-8")).hexdigest(),
            "split": self.split,
            "method": video_dir.parts[-2],
            "video_id": video_dir.name,
            "fps_used": fps_used,
            "bandpass_low": 0.7,
            "bandpass_high": 4.0,
            "chrom_enabled": True,
            "detrend_enabled": True,
            "upsample_enabled": True,
            "spec_size": [self.rppg_size, self.rppg_size],
            "window_policy": "full" if window_len == len(frames) else ("train_random" if self.split == "train" else "fixed_mid"),
            "window_start_frame": 0,
            "window_end_frame": window_len,
            "window_len": window_len,
            "frame_index_basis": "list_order",
            "frame_list_source": frame_source,
            "frame_list_len": len(frames),
            "path_canonicalization": "relative_to_frames_root_forward_slash",
            "frame_list_digest": frame_list_digest,
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
        try:
            with meta_path.open("w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    # ---- rPPG compute ----
    def _compute_rppg(self, frames: Sequence[Path], video_id: str) -> Optional[np.ndarray]:
        fps = self.fps_map.get(video_id, self.rppg_fps)
        min_len = 16

        if self.rppg_window_seconds and self.rppg_window_seconds > 0:
            window_len = int(self.rppg_window_seconds * fps)
        elif self.rppg_window_size and self.rppg_window_size > 0:
            window_len = self.rppg_window_size
        else:
            window_len = len(frames)
        window_len = min(window_len, len(frames))

        if window_len < min_len or len(frames) < min_len:
            return None

        if window_len == len(frames):
            start = 0
        elif self.split == "train":
            start = np.random.randint(0, len(frames) - window_len + 1)
        else:
            center = len(frames) // 2
            start = max(0, center - window_len // 2)
            start = min(start, len(frames) - window_len)
        end = start + window_len
        sel = frames[start:end]

        rgb_ts: List[np.ndarray] = []
        for p in sel:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            mean_rgb = cv2.mean(img)[:3][::-1]  # BGR -> RGB
            rgb_ts.append(mean_rgb)

        if len(rgb_ts) < min_len:
            return None

        rgb_ts_arr = np.array(rgb_ts, dtype=np.float32)
        rgb_ts_arr, eff_fps = self._maybe_upsample_rgb_ts(rgb_ts_arr, fps)
        try:
            _, s_filt, _ = self._chrom_rppg(rgb_ts_arr, fps=eff_fps)
            rppg_img, _ = self._rppg_spectrogram(s_filt, fps=eff_fps, out_size=(self.rppg_size, self.rppg_size))
            return rppg_img
        except Exception:
            return None

    # ---- rPPG processing helpers ----
    def _chrom_rppg(self, rgb_ts: np.ndarray, fps: int, band=(0.7, 4.0)):
        R = rgb_ts[:, 0] / (np.mean(rgb_ts[:, 0]) + 1e-8)
        G = rgb_ts[:, 1] / (np.mean(rgb_ts[:, 1]) + 1e-8)
        B = rgb_ts[:, 2] / (np.mean(rgb_ts[:, 2]) + 1e-8)

        R_d = detrend(R)
        G_d = detrend(G)
        B_d = detrend(B)

        X = 3 * R_d - 2 * G_d
        Y = 1.5 * R_d + G_d - 1.5 * B_d

        alpha = np.std(X) / (np.std(Y) + 1e-8)
        s = X - alpha * Y

        nyq = fps * 0.5
        if nyq <= band[0] * 1.1:
            return s, s, 0.0

        low = band[0] / nyq
        high_hz = min(band[1], nyq * 0.8)
        high = high_hz / nyq
        if high <= low + 1e-3:
            b, a = butter(3, low, btype="high")
            s_f = filtfilt(b, a, s)
            return s, s_f, 0.0

        b, a = butter(3, [low, high], btype="band")
        s_f = filtfilt(b, a, s)
        return s, s_f, 0.0

    def _rppg_spectrogram(
        self,
        s: np.ndarray,
        fps: int,
        nperseg: int = 64,
        noverlap: int = 32,
        out_size: Tuple[int, int] = (128, 128),
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        f, t, Sxx = spectrogram(
            s,
            fs=fps,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="magnitude",
        )
        Sxx = np.abs(Sxx)
        Sxx[Sxx <= 0] = 1e-12
        S_log = 10 * np.log10(Sxx)

        if self.spec_norm_stats is not None:
            mu_f, sigma_f = self.spec_norm_stats
            mu_f = mu_f.reshape(-1, 1)
            sigma_f = sigma_f.reshape(-1, 1)
            S_tilde = (S_log - mu_f) / (sigma_f + 1e-8)
            S_clip = np.clip(S_tilde, -3.0, 3.0)
            S_norm = (S_clip + 3.0) / 6.0
        else:
            cmin, cmax = self.spec_clip
            S_clip = np.clip(S_log, cmin, cmax)
            S_norm = (S_clip - cmin) / (cmax - cmin + 1e-8)

        img = (S_norm * 255).astype(np.uint8)
        img_resized = self._resize_spectrogram_keep_ar(img, out_size)
        return img_resized, (f, t, S_log)

    def _maybe_upsample_rgb_ts(self, rgb_ts: np.ndarray, fps: int):
        if fps >= 6:
            return rgb_ts, fps
        elif fps >= 3:
            target_fps = min(2 * fps, 8)
        else:
            target_fps = 6

        t_orig = np.arange(len(rgb_ts)) / fps
        t_new = np.arange(t_orig[0], t_orig[-1] + 1e-6, 1.0 / target_fps)
        up = []
        for c in range(rgb_ts.shape[1]):
            up_c = np.interp(t_new, t_orig, rgb_ts[:, c])
            up.append(up_c)
        rgb_up = np.stack(up, axis=1).astype(np.float32)
        return rgb_up, target_fps

    def _resize_spectrogram_keep_ar(self, img: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
        tgt_h, tgt_w = target
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((tgt_h, tgt_w), dtype=np.uint8)

        short, long = (h, w) if h < w else (w, h)
        scale = min(tgt_h, tgt_w) / float(short)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((tgt_h, tgt_w), dtype=np.uint8)
        y0 = (tgt_h - new_h) // 2
        x0 = (tgt_w - new_w) // 2
        y1 = y0 + new_h
        x1 = x0 + new_w
        canvas[y0:y1, x0:x1] = resized
        return canvas


if __name__ == "__main__":
    import argparse
    from tqdm.auto import tqdm

    parser = argparse.ArgumentParser(description="Generate rPPG cache + meta for a given split.")
    parser.add_argument("--frames-root", type=str, default="/ssd6/fan/c23", help="Root with split/method/video_id frames.")
    parser.add_argument("--split", type=str, default="train", help="Split to process: train/val/test.")
    parser.add_argument("--rppg-cache", type=str, default="/ssd6/fan/final_project/rppg/runs/rppg_cache", help="Output cache root.")
    parser.add_argument("--frames-per-video", type=int, default=None, help="Optional frame sampling per video (None = all).")
    parser.add_argument("--frame-sample", type=str, default="uniform", help="Sampling strategy if frames_per_video is set.")
    args = parser.parse_args()

    ds = VideoRPPGDatasetWithMeta(
        frames_root=args.frames_root,
        split=args.split,
        rppg_cache_dir=args.rppg_cache,
        frames_per_video=args.frames_per_video,
        frame_sample=args.frame_sample,
    )
    for i in tqdm(range(len(ds)), desc=f"rppg {args.split}"):
        try:
            _ = ds[i]
        except Exception as e:
            print(f"[warn] skip idx {i}: {e}")
