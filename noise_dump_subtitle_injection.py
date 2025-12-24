#!/usr/bin/env python3
"""
Subtitle injection demo (N2 variant): show how subtitles contaminate residual energy.

Given one frame and bbox:
- Aligned ROI (baseline)
- Shifted ROI (dy% downward to eat subtitles)
- Shifted ROI + subtitle strip masked (blur/zero)

Outputs one figure with 3 columns (aligned / shifted / shifted+mask):
- Top row: energy overlay (E = mean_c(|face_ms|), shared color scale via 99th percentile)
- Bottom row: face_ms mosaic (channel-wise normalized, horizontal concat)
Titles include E_sub (subtitle strip) and E_face (rest).
"""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

# Locate project root (contains noise/ and common/)
_FILE = Path(__file__).resolve()
PROJECT_ROOT = None
for anc in _FILE.parents:
    if (anc / "noise").exists() and (anc / "common").exists():
        PROJECT_ROOT = anc
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = _FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.preprocess.preprocess_pipline_v2 import (  # type: ignore
    build_multiscale_residual_stack,
    resize_to,
    to_gray,
)


def clamp_bbox(bbox: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    return (x0, y0, x1, y1)


def shift_bbox(bbox: Tuple[int, int, int, int], dy_frac: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    shift = int(round(dy_frac * h))
    y0 += shift
    y1 += shift
    return clamp_bbox((x0, y0, x1, y1), W, H)


def crop_rgb(img_rgb: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = clamp_bbox(bbox, img_rgb.shape[1], img_rgb.shape[0])
    patch = img_rgb[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros((1, 1, 3), dtype=img_rgb.dtype)
    return patch


def apply_subtitle_mask(gray_crop: np.ndarray, strip: Tuple[float, float], mode: str) -> np.ndarray:
    g = gray_crop.copy()
    H, W = g.shape
    y0 = int(strip[0] * H)
    y1 = int(strip[1] * H)
    y0 = max(0, min(H, y0))
    y1 = max(y0 + 1, min(H, y1))
    if mode == "zero":
        g[y0:y1, :] = 0.0
    else:  # blur
        region = g[y0:y1, :]
        if region.size > 0:
            blurred = cv2.GaussianBlur(region, (0, 0), sigmaX=3.0)
            g[y0:y1, :] = blurred
    return g


def energy_map(face_ms: np.ndarray) -> np.ndarray:
    return np.abs(face_ms).mean(axis=0)


def mosaic(ms: np.ndarray) -> np.ndarray:
    parts = []
    for c in range(ms.shape[0]):
        ch = ms[c]
        ch = ch - ch.min()
        if ch.max() > 1e-6:
            ch = ch / ch.max()
        parts.append(ch)
    return np.concatenate(parts, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True, help="Path to input frame (RGB/BGR image).")
    ap.add_argument("--bbox", type=int, nargs=4, metavar=("x0", "y0", "x1", "y1"), required=True, help="Face bbox.")
    ap.add_argument("--shift-dy", type=float, default=0.35, help="Fraction of bbox height to shift downward.")
    ap.add_argument(
        "--subtitle-strip",
        type=float,
        nargs=2,
        metavar=("y0_rel", "y1_rel"),
        default=(0.80, 1.00),
        help="Relative y-range inside shifted crop to treat as subtitle (e.g., 0.8 1.0).",
    )
    ap.add_argument("--mask-mode", type=str, choices=["blur", "zero"], default="blur", help="Subtitle masking method.")
    ap.add_argument("--ms-size", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("runs_noise/subtitle_injection_demo.png"))
    args = ap.parse_args()

    img_bgr = cv2.imread(str(args.img), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.img}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W, _ = img_rgb.shape
    bbox = clamp_bbox(tuple(args.bbox), W, H)
    bbox_shift = shift_bbox(bbox, args.shift_dy, W, H)

    gray = to_gray(img_rgb)

    def process_crop(bb, mask_subtitle=False):
        crop_gray = crop_rgb(gray[..., None], bb)[..., 0]
        crop_rgb_patch = crop_rgb(img_rgb, bb)
        if mask_subtitle:
            crop_gray = apply_subtitle_mask(crop_gray, args.subtitle_strip, args.mask_mode)
            # also mask rgb for visualization
            y0r = int(args.subtitle_strip[0] * crop_rgb_patch.shape[0])
            y1r = int(args.subtitle_strip[1] * crop_rgb_patch.shape[0])
            y0r = max(0, min(crop_rgb_patch.shape[0], y0r))
            y1r = max(y0r + 1, min(crop_rgb_patch.shape[0], y1r))
            if args.mask_mode == "zero":
                crop_rgb_patch[y0r:y1r, :, :] = 0.0
            else:
                crop_rgb_patch[y0r:y1r, :, :] = cv2.GaussianBlur(crop_rgb_patch[y0r:y1r, :, :], (0, 0), sigmaX=3.0)
        crop_resized = resize_to(crop_gray, args.ms_size)
        ms = build_multiscale_residual_stack(crop_resized)
        e = energy_map(ms)
        return {
            "rgb": cv2.resize(crop_rgb_patch, (args.ms_size, args.ms_size), interpolation=cv2.INTER_AREA),
            "ms": ms,
            "energy": e,
        }

    aligned = process_crop(bbox, mask_subtitle=False)
    shifted = process_crop(bbox_shift, mask_subtitle=False)
    shifted_mask = process_crop(bbox_shift, mask_subtitle=True)

    energies = [aligned["energy"], shifted["energy"], shifted_mask["energy"]]
    all_vals = np.concatenate([e.flatten() for e in energies])
    vmax = float(np.percentile(all_vals, 99))
    if vmax <= 0:
        vmax = 1e-6

    def e_strip_stats(e):
        H, W = e.shape
        y0 = int(args.subtitle_strip[0] * H)
        y1 = int(args.subtitle_strip[1] * H)
        y0 = max(0, min(H, y0))
        y1 = max(y0 + 1, min(H, y1))
        sub = e[y0:y1, :]
        rest_mask = np.ones_like(e, dtype=bool)
        rest_mask[y0:y1, :] = False
        face_vals = e[rest_mask]
        e_sub = float(sub.mean()) if sub.size else 0.0
        e_face = float(face_vals.mean()) if face_vals.size else 0.0
        return e_sub, e_face

    modes = [
        ("Aligned", aligned),
        (f"Shift dy={args.shift_dy:.2f}", shifted),
        (f"Shift+Mask ({args.mask_mode})", shifted_mask),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx, (title, data) in enumerate(modes):
        e_sub, e_face = e_strip_stats(data["energy"])
        # energy overlay
        ax = axes[0, idx]
        ax.imshow(data["rgb"])
        im = ax.imshow(data["energy"], cmap="magma", alpha=0.65, vmin=0.0, vmax=vmax)
        ax.set_title(f"{title}\nE_sub={e_sub:.4f} | E_face={e_face:.4f}")
        ax.axis("off")

        # face_ms mosaic
        ax = axes[1, idx]
        ax.imshow(mosaic(data["ms"]), cmap="gray")
        ax.set_title("face_ms mosaic")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    fig.colorbar(im, cax=cbar_ax, label="Energy (mean |face_ms|)")

    fig.suptitle("Subtitle Injection: residual sensitivity to subtitles", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved figure to {args.out}")


if __name__ == "__main__":
    main()
