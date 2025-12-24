#!/usr/bin/env python3
"""
Simple multimodal training loop using MultiModalDataset + MultiModalDeepfakeModel.

Prerequisites:
 1) rPPG cache + meta produced by rppg/dataset_with_meta.py.
 2) warmup_mm_cache.py has generated plans and filled bbox cache (optional but recommended).

Example:
python scripts/train_multimodal.py \
  --manifest manifest.jsonl \
  --model-path model/resnet-18 \
  --rppg-cache runs/rppg_cache \
  --plan-root runs/mm_plans \
  --bbox-cache bbox_cache
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from common.mm.mm_dataset import MultiModalDataset
from common.mm.multimodal_model import MultiModalDeepfakeModel, multimodal_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSONL.")
    ap.add_argument("--model-path", type=Path, default=Path("model/resnet-18"), help="HF ResNet-18 directory.")
    ap.add_argument(
        "--noise-ckpt",
        type=Path,
        default=Path("/ssd6/fan/final_project/noise/runs_noise_supcon/run_20251212_171816/noise_metric_resnet18_e011_test0.7148.pt"),
        help="Noise branch checkpoint (noise_metric_resnet18*.pt).",
    )
    ap.add_argument(
        "--rppg-ckpt",
        type=Path,
        default=Path("/ssd6/fan/final_project/rppg/runs_rppg/rppg_resnet18_e003_test0.7427.pt"),
        help="rPPG branch checkpoint (rppg_resnet18*.pt).",
    )
    ap.add_argument(
        "--rgb-ckpt",
        type=Path,
        default=Path("/ssd6/fan/final_project/rgb/runs_rgb/best_rgb_resnet18.pt"),
        help="RGB branch checkpoint (best_rgb_resnet18.pt).",
    )
    ap.add_argument("--rppg-cache", type=Path, default=Path("runs/rppg_cache"))
    ap.add_argument("--plan-root", type=Path, default=Path("runs/mm_plans"))
    ap.add_argument("--bbox-cache", type=Path, default=Path("bbox_cache"))
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr-proj", type=float, default=3e-4)
    ap.add_argument("--lr-rppg", type=float, default=3e-5)
    ap.add_argument("--lr-rgb", type=float, default=1e-5)
    ap.add_argument("--lambda-align", type=float, default=0.2)
    ap.add_argument("--lambda-r", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--freeze-rgb", action="store_true", default=True, help="Freeze RGB encoder (default).")
    ap.add_argument("--finetune-rppg", action="store_true", default=True, help="Allow rPPG finetune (default).")
    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate(batch):
    # Drop invalid samples (empty tensors)
    out = []
    for item in batch:
        if (
            item["rgb"].numel() == 0
            or item["noise_face_ms"].numel() == 0
            or item["noise_bg_ms"].numel() == 0
            or item["rppg"].numel() == 0
        ):
            continue
        out.append(item)
    if not out:
        return None
    keys = out[0].keys()
    collated = {}
    for k in keys:
        if k == "meta":
            collated[k] = [b[k] for b in out]
        else:
            collated[k] = torch.stack([b[k] for b in out], dim=0)
    return collated


def build_optimizer(model: MultiModalDeepfakeModel, args):
    params = [
        {
            "params": list(model.proj_noise.parameters())
            + list(model.proj_rppg.parameters())
            + list(model.proj_rgb.parameters())
            + list(model.fusion_head.parameters()),
            "lr": args.lr_proj,
        }
    ]
    if args.finetune_rppg:
        params.append({"params": model.rppg_encoder.parameters(), "lr": args.lr_rppg})
    if not args.freeze_rgb:
        params.append({"params": model.rgb_encoder.parameters(), "lr": args.lr_rgb})
    return AdamW(params)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = MultiModalDataset(
        manifest_path=args.manifest,
        rppg_cache_root=args.rppg_cache,
        plan_root=args.plan_root,
        bbox_cache_root=args.bbox_cache,
        frames_per_video=args.frames_per_video,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    model = MultiModalDeepfakeModel(
        model_path_resnet=args.model_path,
        noise_model_path=args.noise_ckpt,
        rppg_ckpt=args.rppg_ckpt,
        rgb_ckpt=args.rgb_ckpt,
        freeze_noise=True,
        finetune_rppg=args.finetune_rppg,
        freeze_rgb=args.freeze_rgb,
    ).to(device)

    optimizer = build_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"train e{epoch}")
        for batch in pbar:
            if batch is None:
                continue
            optimizer.zero_grad()
            out = model(
                batch["noise_face_ms"].to(device),
                batch["noise_bg_ms"].to(device),
                batch["rppg"].to(device),
                batch["rgb"].to(device),
            )
            loss, logs = multimodal_loss(
                out,
                batch["label"].long().to(device),
                lambda_align=args.lambda_align,
                lambda_r=args.lambda_r,
            )
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "cls": f"{logs['cls']:.4f}", "align": f"{logs['align']:.4f}"})
        scheduler.step()

    print("Training loop finished.")


if __name__ == "__main__":
    main()
