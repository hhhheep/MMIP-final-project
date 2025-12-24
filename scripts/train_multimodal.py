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
import sys
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.mm.mm_dataset import MultiModalDataset
from common.mm.multimodal_model import MultiModalDeepfakeModel, multimodal_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifest.jsonl"),
        help="Path to manifest JSONL (if missing, will auto-scan /ssd6/fan/c23 to create).",
    )
    ap.add_argument("--model-path", type=Path, default=Path("model/resnet-18"), help="HF ResNet-18 directory.")
    ap.add_argument(
        "--noise-ckpt",
        type=Path,
        required=True,
        help="Noise branch checkpoint (noise_metric_resnet18*.pt) — 請自行確認路徑。",
    )
    ap.add_argument(
        "--rppg-ckpt",
        type=Path,
        required=True,
        help="rPPG branch checkpoint (rppg_resnet18*.pt) — 請自行確認路徑。",
    )
    ap.add_argument(
        "--rgb-ckpt",
        type=Path,
        required=True,
        help="RGB branch checkpoint (best_rgb_resnet18.pt) — 請自行確認路徑。",
    )
    ap.add_argument(
        "--rppg-cache",
        type=Path,
        default=Path("runs/rppg_cache"),
        help="Path to rPPG cache+meta directory（請確認符合資料位置）。",
    )
    ap.add_argument("--plan-root", type=Path, default=Path("runs/mm_plans"))
    ap.add_argument("--bbox-cache", type=Path, default=Path("bbox_cache"))
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr-proj", type=float, default=3e-4)
    ap.add_argument("--lr-rppg", type=float, default=3e-5)
    ap.add_argument("--lr-rgb", type=float, default=1e-5)
    ap.add_argument("--lambda-align", type=float, default=0.05)
    ap.add_argument("--lambda-r", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-rgb", action="store_true", default=True, help="Freeze RGB encoder (default).")
    ap.add_argument("--finetune-rppg", action="store_true", default=True, help="Allow rPPG finetune (default).")
    ap.add_argument("--tqdm-ncols", type=int, default=120, help="tqdm bar width; set 0 for auto.")
    ap.add_argument("--eval-only", action="store_true", help="Skip training; run evaluation only.")
    ap.add_argument("--eval-ckpt", type=Path, default=None, help="Checkpoint path for eval-only.")
    ap.add_argument(
        "--ablate",
        type=str,
        default="",
        help="Comma-separated modalities to zero-out during eval (choices: noise,rppg,rgb).",
    )
    ap.add_argument(
        "--allowed-methods",
        type=str,
        default="Real_youtube,Deepfakes,FaceSwap",
        help="Comma-separated methods to include (normalized).",
    )
    ap.add_argument(
        "--train-manips",
        type=str,
        default="Real_youtube,Deepfakes",
        help="Comma-separated manipulation subsets for training.",
    )
    ap.add_argument(
        "--test-manips",
        type=str,
        default="Real_youtube,FaceSwap",
        help="Comma-separated manipulation subsets for val/test.",
    )
    ap.add_argument("--ckpt-out", type=Path, default=Path("runs/mm_ckpt/best_multimodal.pt"), help="Where to save best checkpoint.")
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
        vals = [b[k] for b in out]
        # Non-tensor fields keep為 list
        if k == "meta" or isinstance(vals[0], (str, list, dict)):
            collated[k] = vals
            continue
        if torch.is_tensor(vals[0]):
            collated[k] = torch.stack(vals, dim=0)
        else:
            collated[k] = vals
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

    allowed_methods = set(m.strip() for m in args.allowed_methods.split(",") if m.strip())
    train_methods = set(m.strip() for m in args.train_manips.split(",") if m.strip())
    test_methods = set(m.strip() for m in args.test_manips.split(",") if m.strip())

    # Auto-generate manifest if missing (scan frames_root layout under /ssd6/fan/c23).
    if not args.manifest.exists():
        frames_root = Path("/ssd6/fan/c23")
        print(f"[manifest] {args.manifest} not found; scanning {frames_root} to create one.")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        splits = ["train", "val", "test"]
        with args.manifest.open("w") as f:
            for split in splits:
                split_dir = frames_root / split
                if not split_dir.exists():
                    continue
                for method_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                    method = method_dir.name
                    label = 0 if method.lower().startswith("real") else 1
                    if split == "train":
                        if train_methods and method not in train_methods:
                            continue
                    else:
                        if test_methods and method not in test_methods:
                            continue
                    for vid_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                        rec = {
                            "split": split,
                            "method": method,
                            "video_id": vid_dir.name,
                            "frames_root": str(frames_root),
                            "label": label,
                        }
                        f.write(json.dumps(rec) + "\n")
        print(f"[manifest] written to {args.manifest}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ds = MultiModalDataset(
        manifest_path=args.manifest,
        rppg_cache_root=args.rppg_cache,
        plan_root=args.plan_root,
        bbox_cache_root=args.bbox_cache,
        frames_per_video=args.frames_per_video,
        allowed_methods=train_methods or allowed_methods,
        allowed_splits=["train"],
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

    def build_loader(split_names, methods):
        ds_split = MultiModalDataset(
            manifest_path=args.manifest,
            rppg_cache_root=args.rppg_cache,
            plan_root=args.plan_root,
            bbox_cache_root=args.bbox_cache,
            frames_per_video=args.frames_per_video,
            allowed_methods=methods,
            allowed_splits=split_names,
        )
        if len(ds_split) == 0:
            return None
        return DataLoader(
            ds_split,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )

    # 按需求：train/val 用 train-manips；test 用 test-manips
    val_loader = build_loader(["val"], train_methods)
    test_loader = build_loader(["test"], test_methods)

    def eval_epoch(loader, desc: str):
        model.eval()
        stats = {"loss": 0.0, "cls": 0.0, "align": 0.0, "n": 0}
        logits_all = []
        labels_all = []
        if loader is None:
            return stats
        with torch.no_grad():
            pbar_eval = tqdm(loader, desc=desc, dynamic_ncols=True)
            for batch in pbar_eval:
                if batch is None:
                    continue
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

    best_val = float("inf")
    best_val_auc = -float("inf")
    best_test_auc = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(
            loader,
            desc=f"train e{epoch}",
            dynamic_ncols=True,   # 讓它跟著終端寬度縮放
            mininterval=0.2,
            file=sys.stderr,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",

        )

        # running stats for cosine alignment
        cos_np_sum = 0.0
        cos_nr_sum = 0.0
        cos_pr_sum = 0.0
        n_cos = 0
        loss_sum = 0.0
        cls_sum = 0.0
        align_sum = 0.0
        for batch in pbar:
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
            loss_sum += float(loss.item())
            cls_sum += float(logs["cls"])
            align_sum += float(logs["align"])
            with torch.no_grad():
                s_n = out["s_noise"].detach()
                s_p = out["s_rppg"].detach()
                s_r = out["s_rgb"].detach()
                cos_np = (s_n * s_p).sum(dim=-1)
                cos_nr = (s_n * s_r).sum(dim=-1)
                cos_pr = (s_p * s_r).sum(dim=-1)
                cos_np_b = float(cos_np.mean().item())
                cos_nr_b = float(cos_nr.mean().item())
                cos_pr_b = float(cos_pr.mean().item())
                cos_np_sum += cos_np_b
                cos_nr_sum += cos_nr_b
                cos_pr_sum += cos_pr_b
                n_cos += 1
            loss.backward()
            optimizer.step()
            seen = max(1, n_cos)
            postfix = {
                "loss": f"{loss_sum/seen:.4f}",
                "cls": f"{cls_sum/seen:.4f}",
                "align": f"{align_sum/seen:.4f}",
            }
            pbar.set_postfix(postfix)
            if n_cos % 50 == 0 :
                print(
                    f"[cos] noise-rppg={cos_np_b:.3f} noise-rgb={cos_nr_b:.3f} rppg-rgb={cos_pr_b:.3f}"
                )
        scheduler.step()
        if n_cos > 0:
            print(
                f"[e{epoch:03d}] mean cos: "
                f"np={cos_np_sum/n_cos:.3f} nr={cos_nr_sum/n_cos:.3f} pr={cos_pr_sum/n_cos:.3f}"
            )
        # eval
        val_metrics = eval_epoch(val_loader, desc="val")
        test_metrics = eval_epoch(test_loader, desc="test")
        if val_metrics["n"] > 0:
            print(
                f"[val e{epoch:03d}] loss={val_metrics['loss']:.4f} cls={val_metrics['cls']:.4f} align={val_metrics['align']:.4f} "
                f"acc={val_metrics.get('acc','nan'):.4f} auc={val_metrics.get('auc','nan'):.4f} "
                f"f1={val_metrics.get('f1','nan'):.4f} ap={val_metrics.get('ap','nan'):.4f}"
            )
            # checkpoint rule：val/test AUC > 0.6 且非 NaN
            auc_val = val_metrics.get("auc")
            auc_test = test_metrics.get("auc")
            if (
                auc_val is not None
                and auc_test is not None
                and not np.isnan(auc_val)
                and not np.isnan(auc_test)
                and auc_val > 0.6
                and auc_test > 0.6
            ):
                args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
                # 保存當前 epoch 的合格 checkpoint
                ckpt_name = f"mm_e{epoch:03d}_val{auc_val:.4f}_test{auc_test:.4f}.pt"
                ckpt_path = args.ckpt_out.parent / ckpt_name
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "epoch": epoch,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"[ckpt] saved to {ckpt_path} (val_auc={auc_val:.4f}, test_auc={auc_test:.4f})")
                # 仍保留最佳 val AUC 追蹤
                if auc_val >= best_val_auc:
                    best_val_auc = auc_val
                    best_test_auc = max(best_test_auc, auc_test)
        if test_metrics["n"] > 0:
            print(
                f"[test e{epoch:03d}] loss={test_metrics['loss']:.4f} cls={test_metrics['cls']:.4f} align={test_metrics['align']:.4f} "
                f"acc={test_metrics.get('acc','nan'):.4f} auc={test_metrics.get('auc','nan'):.4f} "
                f"f1={test_metrics.get('f1','nan'):.4f} ap={test_metrics.get('ap','nan'):.4f}"
            )

    print("Training loop finished.")


if __name__ == "__main__":
    main()
