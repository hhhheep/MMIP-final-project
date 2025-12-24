#!/usr/bin/env python3
"""
Evaluation script with modality ablations for the multimodal model.

Example:
python scripts/eval_multimodal_ablate.py \
  --manifest manifest.jsonl \
  --splits val,test \
  --model-path /ssd6/fan/final_project/model/resnet-18 \
  --rppg-cache /ssd6/fan/final_project/rppg/runs/rppg_cache \
  --plan-root runs/mm_plans \
  --bbox-cache bbox_cache \
  --eval-ckpt runs/mm_ckpt/best_multimodal.pt \
  --ablate rgb   # zero out RGB branch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.mm.mm_dataset import MultiModalDataset
from common.mm.multimodal_model import MultiModalDeepfakeModel, multimodal_loss


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate multimodal model with modality ablation.")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSONL.")
    ap.add_argument("--splits", type=str, default="val,test", help="Comma-separated splits to evaluate (e.g., val,test).")
    ap.add_argument("--model-path", type=Path, default=Path("model/resnet-18"), help="HF ResNet-18 directory.")
    ap.add_argument("--noise-ckpt", type=Path, default=Path("/ssd6/fan/final_project/noise/runs_noise_supcon/run_20251212_171816/noise_metric_resnet18_e011_test0.7148.pt"))
    ap.add_argument("--rppg-ckpt", type=Path, default=Path("/ssd6/fan/final_project/rppg/runs_rppg/rppg_resnet18_e003_test0.7427.pt"))
    ap.add_argument("--rgb-ckpt", type=Path, default=Path("/ssd6/fan/final_project/rgb/runs_rgb/best_rgb_resnet18.pt"))
    ap.add_argument("--eval-ckpt", type=Path, default=None, help="Multimodal checkpoint to load (overrides individual ckpts).")
    ap.add_argument("--rppg-cache", type=Path, default=Path("/ssd6/fan/final_project/rppg/runs/rppg_cache"))
    ap.add_argument("--plan-root", type=Path, default=Path("runs/mm_plans"))
    ap.add_argument("--bbox-cache", type=Path, default=Path("bbox_cache"))
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--lambda-align", type=float, default=0.0)
    ap.add_argument("--lambda-r", type=float, default=0.1)
    ap.add_argument(
        "--train-manips",
        type=str,
        default="Real_youtube,Deepfakes",
        help="Manipulations to include for val (if you want same as train).",
    )
    ap.add_argument(
        "--test-manips",
        type=str,
        default="Real_youtube,FaceSwap",
        help="Manipulations to include for test.",
    )
    ap.add_argument(
        "--ablate",
        type=str,
        default="",
        help="Comma-separated modalities to zero-out: choices noise,rppg,rgb",
    )
    return ap.parse_args()


def collate(batch):
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
    collated: Dict[str, List] = {}
    for k in keys:
        vals = [b[k] for b in out]
        if k == "meta" or isinstance(vals[0], (str, list, dict)):
            collated[k] = vals
        elif torch.is_tensor(vals[0]):
            collated[k] = torch.stack(vals, dim=0)
        else:
            collated[k] = vals
    return collated


def eval_epoch(model, loader, device, lambda_align: float, lambda_r: float, ablate_set):
    model.eval()
    stats = {"loss": 0.0, "cls": 0.0, "align": 0.0, "n": 0}
    logits_all = []
    labels_all = []
    if loader is None:
        return stats
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
            if batch is None:
                continue
            noise_face = batch["noise_face_ms"].to(device)
            noise_bg = batch["noise_bg_ms"].to(device)
            rppg = batch["rppg"].to(device)
            rgb = batch["rgb"].to(device)
            if "noise" in ablate_set:
                noise_face = torch.zeros_like(noise_face)
                noise_bg = torch.zeros_like(noise_bg)
            if "rppg" in ablate_set:
                rppg = torch.zeros_like(rppg)
            if "rgb" in ablate_set:
                rgb = torch.zeros_like(rgb)

            out = model(
                noise_face,
                noise_bg,
                rppg,
                rgb,
            )
            loss, logs = multimodal_loss(
                out,
                batch["label"].long().to(device),
                lambda_align=lambda_align,
                lambda_r=lambda_r,
            )
            stats["loss"] += float(loss.item())
            stats["cls"] += float(logs["cls"])
            stats["align"] += float(logs["align"])
            logits_all.append(out["logits"].detach().cpu().numpy())
            labels_all.append(batch["label"].detach().cpu().numpy())
            stats["n"] += 1
    if stats["n"] > 0:
        for k in ["loss", "cls", "align"]:
            stats[k] /= stats["n"]
    if logits_all:
        logits_np = np.concatenate(logits_all, axis=0)
        labels_np = np.concatenate(labels_all, axis=0).astype(int)
        exp_logits = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        probs_class1 = exp_logits[:, 1] / exp_logits.sum(axis=1, keepdims=True)[:, 0]
        preds = np.argmax(logits_np, axis=1)
        if np.unique(labels_np).size < 2:
            stats.update(
                {
                    "acc": accuracy_score(labels_np, preds),
                    "f1": f1_score(labels_np, preds, average="macro", zero_division=0.0),
                    "auc": float("nan"),
                    "ap": float("nan"),
                }
            )
        else:
            stats.update(
                {
                    "acc": balanced_accuracy_score(labels_np, preds),
                    "f1": f1_score(labels_np, preds, average="macro", zero_division=0.0),
                    "auc": roc_auc_score(labels_np, probs_class1),
                    "ap": average_precision_score(labels_np, probs_class1),
                }
            )
    return stats


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ablate_set = set(m.strip().lower() for m in args.ablate.split(",") if m.strip())

    train_methods = set(m.strip() for m in args.train_manips.split(",") if m.strip())
    test_methods = set(m.strip() for m in args.test_manips.split(",") if m.strip())
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    ds = MultiModalDataset(
        manifest_path=args.manifest,
        rppg_cache_root=args.rppg_cache,
        plan_root=args.plan_root,
        bbox_cache_root=args.bbox_cache,
        frames_per_video=args.frames_per_video,
        allowed_methods=test_methods if "test" in splits else train_methods,
        allowed_splits=splits,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
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
        finetune_rppg=False,
        freeze_rgb=True,
    ).to(device)

    # 如果提供了多模態 ckpt，直接載入
    if args.eval_ckpt and args.eval_ckpt.exists():
        state = torch.load(args.eval_ckpt, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        print(f"[load] loaded {args.eval_ckpt}")

    stats = eval_epoch(model, loader, device, args.lambda_align, args.lambda_r, ablate_set)
    print(
        f"[eval splits={splits} ablate={','.join(sorted(ablate_set)) or 'none'}] "
        f"loss={stats.get('loss','nan'):.4f} cls={stats.get('cls','nan'):.4f} align={stats.get('align','nan'):.4f} "
        f"acc={stats.get('acc','nan'):.4f} auc={stats.get('auc','nan'):.4f} f1={stats.get('f1','nan'):.4f} ap={stats.get('ap','nan'):.4f}"
    )


if __name__ == "__main__":
    main()
