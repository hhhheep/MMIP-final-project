

# ============================================================
# Imports
# ============================================================
import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List, Sequence, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from scipy.signal import detrend, butter, filtfilt, spectrogram, welch

# ============================================================
# New: Video-level rPPG Dataset (直接從 frame 生成代表整支影片的 rPPG)
# 參考 noise_face_dataset_v2 的掃描方式，不依賴 rppg_path。
# ============================================================
class VideoRPPGDataset(Dataset):
    """
    從已裁好的影片幀序列生成單張 rPPG 時頻圖，代表整支影片。
    目錄假設: root/<split>/<method>/<video_id>/*.png
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

        self.rppg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.samples = self._build_index()

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

    def _list_frames(self, video_dir: Path) -> List[Path]:
        # 優先依照 list.txt 指定的順序（每行為檔名）
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
                    return ordered
            except Exception:
                pass  # fallback to glob
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            frames = sorted(video_dir.glob(pattern))
            if frames:
                return frames
        return []

    def _maybe_sample_frames(self, frames: Sequence[Path], rel_video: str) -> List[Path]:
        # 若 frames_per_video 為 None，使用全部幀
        if self.frames_per_video is None or len(frames) <= self.frames_per_video:
            return list(frames)
        if self.frame_sample == "random":
            idxs = np.random.choice(len(frames), size=self.frames_per_video, replace=False)
            return [frames[i] for i in sorted(idxs)]
        # default uniform
        idxs = np.linspace(0, len(frames) - 1, self.frames_per_video, dtype=int)
        return [frames[i] for i in idxs]

    def __getitem__(self, idx: int):
        video_dir, label, method = self.samples[idx]
        frames = self._list_frames(video_dir)
        if not frames:
            raise FileNotFoundError(f"No frames found in {video_dir}")
        frames = self._maybe_sample_frames(frames, rel_video=str(video_dir.relative_to(self.frames_root)))

        cache_img = self._load_cache(video_dir)
        if cache_img is None:
            cache_img = self._compute_rppg(frames, video_dir.name)
            if cache_img is None:
                # 計算失敗則跳過該樣本
                raise IndexError("rPPG computation failed; skip sample")
            self._save_cache(video_dir, cache_img)

        I_rppg = self.rppg_transform(Image.fromarray(cache_img).convert("L"))
        return I_rppg, torch.tensor(float(label))

    # ---- cache helpers ----
    def _cache_key(self, video_dir: Path) -> str:
        h = hashlib.md5()
        h.update(str(video_dir).encode("utf-8"))
        h.update(f"{self.rppg_window_size}_{self.rppg_fps}".encode("utf-8"))
        return h.hexdigest() + ".png"

    def _load_cache(self, video_dir: Path) -> Optional[np.ndarray]:
        if not self.rppg_cache_dir:
            return None
        cache_path = self.rppg_cache_dir / self.split / video_dir.parts[-2] / video_dir.name / self._cache_key(video_dir)
        if cache_path.exists():
            try:
                img = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
            except Exception:
                return None
        return None

    def _save_cache(self, video_dir: Path, img: np.ndarray):
        if not self.rppg_cache_dir:
            return
        cache_path = self.rppg_cache_dir / self.split / video_dir.parts[-2] / video_dir.name / self._cache_key(video_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cv2.imwrite(str(cache_path), img)
        except Exception:
            pass

    # ---- rPPG compute ----
    def _compute_rppg(self, frames: Sequence[Path], video_id: str) -> Optional[np.ndarray]:
        fps = self.fps_map.get(video_id, self.rppg_fps)
        min_len = 16  # 比 8 更嚴格的最短序列長度

        # 決定窗口長度（優先秒數，再來固定幀數；0/None 表示全長）
        if self.rppg_window_seconds and self.rppg_window_seconds > 0:
            window_len = int(self.rppg_window_seconds * fps)
        elif self.rppg_window_size and self.rppg_window_size > 0:
            window_len = self.rppg_window_size
        else:
            window_len = len(frames)
        window_len = min(window_len, len(frames))

        if window_len < min_len or len(frames) < min_len:
            return None

        # 視窗選擇：train 隨機，val/test 取中間；都保持時間順序
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
        # FPS 太低，無法支援 band 下限，直接略過 band-pass
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

    def _resize_spectrogram_keep_ar(self, img: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
        """
        等比例縮放，短邊對齊 target_short，長邊做 average pool or zero-pad 到 target_long。
        預設 target=(H,W) = (rppg_size, rppg_size) e.g. 224x224。
        """
        tgt_h, tgt_w = target
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((tgt_h, tgt_w), dtype=np.uint8)

        short, long = (h, w) if h < w else (w, h)
        scale = min(tgt_h, tgt_w) / float(short)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 長邊處理：若超過目標，做平均池化；若不足，做 zero pad（置中）
        out = img_resized
        if new_h > tgt_h:
            k = new_h // tgt_h
            if k > 1:
                out = cv2.blur(out, (1, k))[::k, :]
                new_h = out.shape[0]
        if new_w > tgt_w:
            k = new_w // tgt_w
            if k > 1:
                out = cv2.blur(out, (k, 1))[:, ::k]
                new_w = out.shape[1]

        pad_h = max(0, tgt_h - out.shape[0])
        pad_w = max(0, tgt_w - out.shape[1])
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        out_padded = cv2.copyMakeBorder(out, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return out_padded

    def _rppg_spectrogram(self, rppg: np.ndarray, fps: int, out_size: Tuple[int, int] = (224, 224)):
        input_length = len(rppg)
        if input_length < 8:
            raise ValueError("rPPG sequence too short for spectrogram")

        nperseg = 64 if input_length >= 64 else max(8, input_length // 2)
        nperseg = min(nperseg, input_length)
        noverlap = nperseg // 2

        f, t, Sxx = spectrogram(rppg, fs=fps, nperseg=nperseg, noverlap=noverlap)
        S_log = np.log10(Sxx + 1e-8)

        # freq-wise normalize if stats provided
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
        """依 fps 自適應插值：>=6Hz 不動，3-6Hz 插值到 2x，<3Hz 插值到 6Hz。"""
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
            # 可選：先做輕微平滑，這裡直接線性插值
            up_c = np.interp(t_new, t_orig, rgb_ts[:, c])
            up.append(up_c)
        rgb_up = np.stack(up, axis=1).astype(np.float32)
        return rgb_up, target_fps
