#!/usr/bin/env python3
"""
Generate a 2x2 rPPG evidence pack (R0) for a single video:
Mean frame | time-series (s_raw vs s_filt) | PSD with peak | Spectrogram with HR track.
Defaults: 8s window, fps from csv map if available else 30, bandpass on, no shuffle.

This script is intentionally lightweight so R1/R2 abl ablations can reuse the flags:
--bandpass {on,off}, --time-shuffle {on,off}.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, filtfilt, medfilt, spectrogram, welch


def read_fps_map(fps_csv: Path | None) -> Dict[str, float]:
    if not fps_csv or not fps_csv.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(fps_csv, header=None, names=["video_id", "fps"], index_col=0, converters={0: str})
        return df["fps"].to_dict()
    except Exception:
        return {}


def list_frames(video_dir: Path) -> List[Path]:
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
            pass
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        frames = sorted(video_dir.glob(pattern))
        if frames:
            return frames
    return []


def select_window(frames: Sequence[Path], fps: float, window_seconds: float) -> Tuple[List[Path], int, int]:
    window_len = int(round(window_seconds * fps))
    window_len = max(1, min(window_len, len(frames)))
    if window_len < 16:
        raise ValueError(f"Not enough frames for window: have {len(frames)}, need >=16")
    center = len(frames) // 2
    start = max(0, center - window_len // 2)
    start = min(start, len(frames) - window_len)
    end = start + window_len
    return list(frames[start:end]), start, end


def compute_rgb_ts(frames: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    rgb_ts: List[np.ndarray] = []
    acc = None
    used: List[Path] = []
    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if acc is None:
            acc = img_rgb.astype(np.float64)
        else:
            acc += img_rgb
        mean_rgb = img_rgb.reshape(-1, 3).mean(axis=0)
        rgb_ts.append(mean_rgb)
        used.append(p)
    if acc is None or not rgb_ts:
        raise ValueError("Failed to read any frames")
    mean_frame = (acc / len(used)).astype(np.uint8)
    return np.stack(rgb_ts, axis=0).astype(np.float32), mean_frame, used


def chrom_rppg(rgb_ts: np.ndarray, fps: float, band: Tuple[float, float], bandpass: bool = True):
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

    if not bandpass:
        return s, s

    nyq = fps * 0.5
    if nyq <= band[0] * 1.1:
        return s, s

    low = band[0] / nyq
    high_hz = min(band[1], nyq * 0.8)
    high = high_hz / nyq
    if high <= low + 1e-3:
        b, a = butter(3, low, btype="high")
        return s, filtfilt(b, a, s)

    b, a = butter(3, [low, high], btype="band")
    return s, filtfilt(b, a, s)


def maybe_shuffle(ts: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ts))
    rng.shuffle(idx)
    return ts[idx]


def hr_track_from_spec(f: np.ndarray, t: np.ndarray, Sxx: np.ndarray, band: Tuple[float, float]):
    band_mask = (f >= band[0]) & (f <= band[1])
    f_band = f[band_mask]
    if f_band.size == 0:
        return np.array([]), np.array([])
    S_band = Sxx[band_mask, :]
    peak_idx = np.argmax(S_band, axis=0)
    peak_f = f_band[peak_idx]
    peak_f_smooth = medfilt(peak_f, kernel_size=3) if peak_f.size >= 3 else peak_f
    return peak_f, peak_f_smooth


def plot_evidence(mean_frame: np.ndarray, t_axis: np.ndarray, s_raw: np.ndarray, s_filt: np.ndarray,
                  f_psd: np.ndarray, psd: np.ndarray, peak_hz: float,
                  f_spec: np.ndarray, t_spec: np.ndarray, Sxx: np.ndarray,
                  hr_track: np.ndarray, hr_track_smooth: np.ndarray, band: Tuple[float, float],
                  fps_used: float, window_seconds: float, out_path: Path):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].imshow(mean_frame)
    axs[0, 0].set_title("Mean frame")
    axs[0, 0].axis("off")
    axs[0, 0].text(0.02, 0.95, f"fps_used={fps_used:.2f}\nwindow={window_seconds:.1f}s",
                   transform=axs[0, 0].transAxes, fontsize=9, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    axs[0, 1].plot(t_axis, s_raw, label="s_raw", alpha=0.7)
    axs[0, 1].plot(t_axis, s_filt, label="s_filt", alpha=0.9)
    axs[0, 1].set_title("rPPG time-series")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].legend()

    axs[1, 0].plot(f_psd, psd)
    axs[1, 0].axvspan(band[0], band[1], color="orange", alpha=0.1, label="band")
    axs[1, 0].axvline(peak_hz, color="r", linestyle="--", label=f"peak={peak_hz:.2f}Hz ({peak_hz*60:.1f} BPM)")
    axs[1, 0].set_xlim(0, max(5.0, band[1] + 0.5))
    axs[1, 0].set_title("PSD (Welch)")
    axs[1, 0].set_xlabel("Freq (Hz)")
    axs[1, 0].set_ylabel("Power")
    axs[1, 0].legend()

    S_log = 10 * np.log10(Sxx + 1e-12)
    im = axs[1, 1].pcolormesh(t_spec, f_spec, S_log, shading="auto", cmap="magma")
    axs[1, 1].plot(t_spec, hr_track, color="cyan", alpha=0.5, linewidth=1.2, label="peak f")
    axs[1, 1].plot(t_spec, hr_track_smooth, color="lime", alpha=0.9, linewidth=1.5, label="peak f (medfilt)")
    axs[1, 1].set_ylim(0, max(5.0, band[1] + 0.5))
    axs[1, 1].set_title("Spectrogram + HR track")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Freq (Hz)")
    axs[1, 1].legend()
    cbar = fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Power (dB)")

    fig.suptitle("rPPG evidence pack (R0)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate rPPG R0 evidence pack (2x2 PNG + meta).")
    ap.add_argument("--frames-root", type=Path, required=True, help="Root containing split/method/video_id frames.")
    ap.add_argument("--split", type=str, required=True, help="Split name, e.g., val or test.")
    ap.add_argument("--method", type=str, required=True, help="Method dir under split.")
    ap.add_argument("--video-id", type=str, required=True, help="Video id dir under method.")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/rppg_viz"), help="Where to save PNG/meta.")
    ap.add_argument("--window-seconds", type=float, default=8.0, help="Window length in seconds.")
    ap.add_argument("--fps-csv", type=Path, default=Path("/ssd6/fan/c23/bkChen/MMIP/data/fps_list.csv"))
    ap.add_argument("--bandpass", choices=["on", "off"], default="on", help="Toggle bandpass filter.")
    ap.add_argument("--time-shuffle", choices=["on", "off"], default="off", help="Shuffle time order of RGB ts.")
    ap.add_argument("--seed", type=int, default=1337, help="Seed for shuffle.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    fps_map = read_fps_map(args.fps_csv)
    video_dir = args.frames_root / args.split / args.method / args.video_id
    if not video_dir.exists():
        raise FileNotFoundError(f"Video dir not found: {video_dir}")

    frames = list_frames(video_dir)
    if not frames:
        raise FileNotFoundError(f"No frames found in {video_dir}")

    fps_used = float(fps_map.get(args.video_id, 30.0))
    window_frames, start, end = select_window(frames, fps_used, args.window_seconds)
    rgb_ts, mean_frame, used_frames = compute_rgb_ts(window_frames)

    if args.time_shuffle == "on":
        rgb_ts_proc = maybe_shuffle(rgb_ts, seed=args.seed)
    else:
        rgb_ts_proc = rgb_ts

    band = (0.7, 4.0)
    s_raw, s_filt = chrom_rppg(rgb_ts_proc, fps=fps_used, band=band, bandpass=args.bandpass == "on")
    t_axis = np.arange(len(s_raw)) / fps_used

    nperseg = min(len(s_filt), 256)
    nperseg = max(16, nperseg)
    noverlap = nperseg // 2
    f_psd, psd = welch(s_filt, fs=fps_used, nperseg=nperseg, noverlap=noverlap)
    band_mask = (f_psd >= band[0]) & (f_psd <= band[1])
    if not np.any(band_mask):
        peak_hz = float("nan")
    else:
        peak_idx = np.argmax(psd[band_mask])
        peak_hz = f_psd[band_mask][peak_idx]

    f_spec, t_spec, Sxx = spectrogram(
        s_filt,
        fs=fps_used,
        window="hann",
        nperseg=min(len(s_filt), 64) if len(s_filt) >= 64 else max(8, len(s_filt) // 2),
        noverlap=None,
        scaling="density",
        mode="magnitude",
    )
    hr_track, hr_track_smooth = hr_track_from_spec(f_spec, t_spec, Sxx, band=band)

    out_base = args.out_dir / f"{args.split}_{args.method}_{args.video_id}_R0"
    out_png = out_base.with_suffix(".png")
    plot_evidence(
        mean_frame,
        t_axis,
        s_raw,
        s_filt,
        f_psd,
        psd,
        peak_hz,
        f_spec,
        t_spec,
        Sxx,
        hr_track,
        hr_track_smooth,
        band,
        fps_used,
        args.window_seconds,
        out_png,
    )

    meta = {
        "frames_root": str(args.frames_root),
        "split": args.split,
        "method": args.method,
        "video_id": args.video_id,
        "fps_used": fps_used,
        "window_seconds": args.window_seconds,
        "band": list(band),
        "bandpass": args.bandpass,
        "time_shuffle": args.time_shuffle,
        "seed": args.seed,
        "window_start_idx": start,
        "window_end_idx": end,
        "frames_used": [p.relative_to(args.frames_root).as_posix() for p in used_frames],
        "peak_hz": peak_hz,
        "peak_bpm": peak_hz * 60 if not math.isnan(peak_hz) else float("nan"),
        "hr_track_smoothed": hr_track_smooth.tolist() if hr_track_smooth.size else [],
    }
    out_meta = out_base.with_suffix(".meta.json")
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved PNG: {out_png}")
    print(f"Saved meta: {out_meta}")


if __name__ == "__main__":
    main()
