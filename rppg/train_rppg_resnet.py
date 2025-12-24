#!/usr/bin/env python3
"""
Training script: rPPG spectrogram ResNet-18 (1-channel) built on VideoRPPGDataset.

參考 train_rgb_resnet.py 的流程與指標，直接從 frames_root/<split>/<method>/<video_id>/*.png
生成單張 rPPG 時頻圖代表影片。可使用快取避免重算。
"""

import argparse
import random
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import ResNetForImageClassification
from tqdm.auto import tqdm

from rppg.dataset import VideoRPPGDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--frames-root",
        type=Path,
        default=Path("/ssd6/fan/c23"),
        help="裁剪好幀的根目錄，預設 /ssd6/fan/c23",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("./runs_rppg"))
    ap.add_argument("--model-path", type=Path, default=Path("/ssd6/fan/final_project/model/resnet-18"), help="本地 HuggingFace ResNet-18 目錄")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6, help="minimum LR for cosine annealing")
    ap.add_argument("--warmup-epochs", type=int, default=2, help="linear warmup epochs before cosine schedule")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--frames-per-video", type=int, default=None, help="None 表示使用影片全部幀")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train-manips", type=str, default="Real_youtube,Deepfakes")
    ap.add_argument("--test-manips", type=str, default="Real_youtube,FaceSwap")
    ap.add_argument("--rppg-cache-dir", type=Path, default=Path("runs/rppg_cache"))
    ap.add_argument("--rppg-window-size", type=int, default=0, help="0 或未設置即使用全部幀")
    ap.add_argument("--rppg-fps", type=int, default=30, help="無 fps 表時的預設 fps")
    ap.add_argument("--fps-csv", type=Path, default=Path("/ssd6/fan/c23/bkChen/MMIP/data/fps_list.csv"),
                    help="video_id,fps 對應表，用於正確頻率刻度")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def _manip_list(val: str):
    return [m.strip() for m in val.split(",") if m.strip()]


def build_dataloaders(args):
    train_manips = _manip_list(args.train_manips)
    test_manips = _manip_list(args.test_manips)
    fps_map = None
    if args.fps_csv and args.fps_csv.exists():
        try:
            import pandas as pd
            df_fps = pd.read_csv(args.fps_csv, header=None, names=["video_id", "fps"], index_col=0, converters={0: str})
            fps_map = df_fps["fps"].to_dict()
        except Exception:
            fps_map = None

    train_ds = VideoRPPGDataset(
        frames_root=args.frames_root,
        split="train",
        frames_per_video=args.frames_per_video,
        frame_sample="uniform",  # 若 frames_per_video 非 None 則均勻取樣；None 時用全部幀
        allowed_methods=train_manips,
        rppg_cache_dir=args.rppg_cache_dir,
        rppg_window_size=args.rppg_window_size,
        rppg_fps=args.rppg_fps,
        fps_map=fps_map,
    )
    val_ds = VideoRPPGDataset(
        frames_root=args.frames_root,
        split="val",
        frames_per_video=args.frames_per_video,
        frame_sample="fixed",  # 若 frames_per_video 非 None 則固定取樣；None 時用全部幀
        allowed_methods=train_manips,
        rppg_cache_dir=args.rppg_cache_dir,
        rppg_window_size=args.rppg_window_size,
        rppg_fps=args.rppg_fps,
        fps_map=fps_map,
    )
    test_ds = VideoRPPGDataset(
        frames_root=args.frames_root,
        split="test",
        frames_per_video=args.frames_per_video,
        frame_sample="fixed",  # 若 frames_per_video 非 None 則固定取樣；None 時用全部幀
        allowed_methods=test_manips,
        rppg_cache_dir=args.rppg_cache_dir,
        rppg_window_size=args.rppg_window_size,
        rppg_fps=args.rppg_fps,
        fps_map=fps_map,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


class RPPGResNet18(nn.Module):
    """HF ResNet-18 1-channel head (2 logits)，從本地模型載入。"""

    def __init__(self, model_path: Path):
        super().__init__()
        base = ResNetForImageClassification.from_pretrained(
            model_path,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        self.net = base

    def forward(self, x):
        x3 = x.repeat(1, 3, 1, 1)  # 將 1 channel rPPG 複製成 3 channel 以符合預訓練 ResNet 輸入
        return self.net(x3).logits


def build_warmup_cosine_scheduler(optimizer, args):
    """Linear warmup -> cosine annealing to lr_min."""
    base_lr = args.lr
    eta_min = args.lr_min
    warmup = max(0, args.warmup_epochs)
    total = max(1, args.epochs)

    def lr_lambda(epoch: int):
        if epoch < warmup:
            return float(epoch + 1) / float(max(1, warmup))
        t = (epoch - warmup) / float(max(1, total - warmup))
        cos_term = 0.5 * (1.0 + math.cos(math.pi * t))
        return (eta_min / base_lr) + cos_term * (1.0 - eta_min / base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_samples = 0
    running_correct = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for I_rppg, labels in pbar:
        x = I_rppg.to(device)  # (B,1,128,128)
        y = labels.to(device).long()

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == y).sum().item()
        avg_loss = total_loss / max(1, n_samples)
        avg_acc = running_correct / max(1, n_samples)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})

    return total_loss / max(1, n_samples)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    pbar = tqdm(loader, desc="eval", leave=False)
    for I_rppg, labels in pbar:
        x = I_rppg.to(device)
        y = labels.to(device).long()
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    if not all_labels:
        return {"acc": 0.0, "auc": 0.0, "f1": 0.0, "ap": 0.0}

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_class1 = exp_logits[:, 1] / exp_logits.sum(axis=1, keepdims=True)[:, 0]
    preds = np.argmax(logits, axis=1)

    if np.unique(labels).size < 2:
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0.0)
        auc = float("nan")
        ap = float("nan")
    else:
        acc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0.0)
        auc = roc_auc_score(labels, probs_class1)
        ap = average_precision_score(labels, probs_class1)

    return {"acc": acc, "auc": auc, "f1": f1, "ap": ap}


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.out_dir / f"train_log_{run_ts}.txt"

    def write_log(msg: str):
        print(msg)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = RPPGResNet18(model_path=args.model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, args)

    best_val_auc = 0.0
    best_test_auc = 0.0
    last_saved_ckpt = None
    run_dir = args.out_dir

    def _fmt(x: float) -> str:
        if x != x:
            return "nan"
        return f"{x:.4f}"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, device)
        test_metrics = eval_epoch(model, test_loader, device)
        scheduler.step()

        log_line = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val acc={_fmt(val_metrics['acc'])} auc={_fmt(val_metrics['auc'])} "
            f"f1={_fmt(val_metrics['f1'])} ap={_fmt(val_metrics['ap'])} | "
            f"other acc={_fmt(test_metrics['acc'])} auc={_fmt(test_metrics['auc'])} "
            f"f1={_fmt(test_metrics['f1'])} ap={_fmt(test_metrics['ap'])}"
        )
        write_log(log_line)

        should_save = (
            not math.isnan(val_metrics["auc"])
            and not math.isnan(test_metrics["auc"])
            and val_metrics["auc"] > 0.6
            and test_metrics["auc"] > 0.6
        )
        if should_save:
            best_val_auc = max(best_val_auc, val_metrics["auc"])
            best_test_auc = max(best_test_auc, test_metrics["auc"])
            ckpt_name = f"rppg_resnet18_e{epoch:03d}_test{test_metrics['auc']:.4f}.pt"
            ckpt_path_epoch = run_dir / ckpt_name
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": val_metrics["auc"],
                    "test_auc": test_metrics["auc"],
                    "args": vars(args),
                },
                ckpt_path_epoch,
            )
            last_saved_ckpt = ckpt_path_epoch
            write_log(f"  ↳ Saved checkpoint {ckpt_name} (val_auc={val_metrics['auc']:.4f}, test_auc={test_metrics['auc']:.4f})")

    # if last_saved_ckpt is None:
    #     write_log("No checkpoint met saving criteria; skip final reload/eval.")
    # else:
    #     ckpt = torch.load(last_saved_ckpt, map_location=device)
    #     model.load_state_dict(ckpt["model_state"])
    #     test_metrics = eval_epoch(model, test_loader, device)
    #     write_log(
    #         f"[TEST] acc={_fmt(test_metrics['acc'])} auc={_fmt(test_metrics['auc'])} "
    #         f"f1={_fmt(test_metrics['f1'])} ap={_fmt(test_metrics['ap'])}"
    #     )


if __name__ == "__main__":
    main()
