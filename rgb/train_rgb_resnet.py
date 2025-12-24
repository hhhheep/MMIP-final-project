#!/usr/bin/env python3
"""
Baseline training script: RGB ResNet-18 on DeepfakeFaceDataset.

Features:
- Supports cross-subset configs via --train-manips / --test-manips.
- Uses ImageNet-pretrained ResNet-18 with 1-logit head (BCEWithLogits).
- Simple cosine LR schedule and checkpoint on best val AUC.
"""

import argparse
import random
import sys
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
from torchvision import transforms
from transformers import ResNetForImageClassification
from tqdm.auto import tqdm

from common.datasets.deepfake_dataset import DeepfakeFaceDataset

DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "resnet-18"
DEFAULT_OUT_DIR = HERE / "runs_rgb"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True, help="FF++ c23 root, e.g., /ssd2/deepfake/c23")
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="HuggingFace ResNet-18 directory")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train-manips", type=str, default="Real,Deepfakes")
    # default test manips: include more fakes for harder eval
    ap.add_argument("--test-manips", type=str, default="Real,Deepfakes,FaceSwap,Face2Face,NeuralTextures")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def build_dataloaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _manip_list(val: str):
        return [m.strip() for m in val.split(",") if m.strip()]

    train_manips = _manip_list(args.train_manips)
    test_manips = _manip_list(args.test_manips)

    train_ds = DeepfakeFaceDataset(
        data_root=args.data_root,
        split="train",
        frames_per_video=args.frames_per_video,
        frame_sample="uniform",
        allowed_manips=train_manips,
        crop_size=224,
        transform=normalize,
    )
    val_ds = DeepfakeFaceDataset(
        data_root=args.data_root,
        split="val",
        frames_per_video=args.frames_per_video,
        frame_sample="uniform",
        allowed_manips=train_manips,
        crop_size=224,
        transform=normalize,
    )
    test_ds = DeepfakeFaceDataset(
        data_root=args.data_root,
        split="test",
        frames_per_video=args.frames_per_video,
        frame_sample="uniform",
        allowed_manips=test_manips,
        crop_size=224,
        transform=normalize,
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


class RGBResNet18HF(nn.Module):
    """HuggingFace ResNet-18 (convnet, not ViT) with 2-logit head."""

    def __init__(self, model_path: Path):
        super().__init__()
        # ignore_mismatched_sizes to swap 1000-way head -> 2
        self.backbone = ResNetForImageClassification.from_pretrained(
            model_path, num_labels=2, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        logits = self.backbone(x).logits  # (B,2)
        return logits


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device)  # (B,3,224,224)
        y = batch["label"].to(device).long()  # (B,)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device).long()
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    if not all_labels:
        return {"acc": 0.0, "auc": 0.0, "f1": 0.0, "ap": 0.0}

    logits = np.concatenate(all_logits)  # (N,2)
    labels = np.concatenate(all_labels)

    # softmax probability for class 1 (fake)
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

    # fix randomness for reproducibility (within dataloader-worker variance)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(args)

    model = RGBResNet18HF(model_path=args.model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0

    def _fmt(x: float) -> str:
        if x != x:  # NaN check
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

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            ckpt_path = args.out_dir / "best_rgb_resnet18.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_auc": val_metrics["auc"],
                    "args": vars(args),
                },
                ckpt_path,
            )
            write_log(f"  ↳ New best AUC, checkpoint saved to {ckpt_path}")

    # test with best checkpoint
    ckpt = torch.load(args.out_dir / "best_rgb_resnet18.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = eval_epoch(model, test_loader, device)
    write_log(
        f"[TEST] acc={_fmt(test_metrics['acc'])} auc={_fmt(test_metrics['auc'])} "
        f"f1={_fmt(test_metrics['f1'])} ap={_fmt(test_metrics['ap'])}"
    )


if __name__ == "__main__":
    main()
