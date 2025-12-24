#!/usr/bin/env python3
"""
Noise-focused Siamese training with HF ResNet-18 backbone.

Flow:
- Use NoiseFaceDataset to load face_ms / bg_ms tensors (3, ms, ms).
- Shared HF ResNet-18 backbone extracts feature maps without pooling.
- Compute cosine similarity matrix between face/bg spatial features.
- Tiny CNN head outputs 2 logits (real/fake) on the similarity map.

Logging:
- In-domain (val) and cross-subset (test) evaluated every epoch.
- Metrics: acc, auc, f1, ap. AUC/AP become nan if split is single-class.
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import ResNetModel

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "c23"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "resnet-18"
DEFAULT_OUT_DIR = HERE / "runs_noise_supcon"

# 使用 v2 版資料集（包含改進的採樣/快取邏輯）
from noise.datasets.noise_face_dataset_v2 import NoiseFaceDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="FF++ c23 root, e.g., /ssd6/fan/c23",
    )
    ap.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="HuggingFace ResNet-18 directory")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6, help="minimum LR for cosine annealing")
    ap.add_argument("--warmup-epochs", type=int, default=2, help="linear warmup epochs before cosine schedule")
    ap.add_argument("--contrastive-weight", type=float, default=0.0, help="(deprecated) margin weight; kept for compatibility")
    ap.add_argument("--contrastive-margin", type=float, default=0.5, help="(deprecated) margin m")
    ap.add_argument("--margin-scale", type=float, default=5.0, help="(deprecated) margin slope")
    ap.add_argument("--supcon-weight", type=float, default=0.02, help="lambda for supervised contrastive loss on feat_diff")
    ap.add_argument("--supcon-temperature", type=float, default=0.2, help="temperature tau for supervised contrastive loss")

    # --- ROI self-consistency regularizers ---
    ap.add_argument("--roi-gap-weight", type=float, default=0.02, help="lambda for batch-wise ROI mean-gap loss")
    ap.add_argument("--roi-gap-margin", type=float, default=0.70, help="target margin for (mu_fake - mu_real) on dist")
    ap.add_argument("--roi-rank-weight", type=float, default=0.01, help="lambda for uncertain-region ROI ranking loss")
    ap.add_argument("--roi-rank-margin", type=float, default=0.3, help="margin m_pair in ranking loss: r_fake - r_real >= m_pair")
    ap.add_argument("--roi-rank-pmin", type=float, default=0.30, help="lower prob threshold for 'uncertain' samples")
    ap.add_argument("--roi-rank-pmax", type=float, default=0.70, help="upper prob threshold for 'uncertain' samples")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--ms-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train-manips", type=str, default="Real_youtube,Deepfakes")
    # harder cross-subset by default
    ap.add_argument("--test-manips", type=str, default="Real_youtube,FaceSwap")
    return ap.parse_args()


def _manip_list(val: str) -> List[str]:
    return [m.strip() for m in val.split(",") if m.strip()]


def build_noise_loaders(args):
    train_manips = _manip_list(args.train_manips)
    test_manips = _manip_list(args.test_manips)

    def make_ds(split: str, manips: List[str], frame_sample: str = "uniform"):
        return NoiseFaceDataset(
            data_root=args.data_root,
            split=split,
            frames_per_video=args.frames_per_video,
            frame_sample=frame_sample,
            allowed_manips=manips,
            crop_size=224,
            ms_size=args.ms_size,
            include_rgb=False,
            include_ms_face=True,
            include_ms_bg=True,
            include_residual=False,
            precompute_dir=args.data_root / "proc_cache",
        )

    train_ds = make_ds("train", train_manips, frame_sample="uniform")
    val_ds = make_ds("val", train_manips, frame_sample="fixed")
    test_ds = make_ds("val", test_manips, frame_sample="fixed")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


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


class PositiveDistHead(nn.Module):
    """Distance head with strictly non-negative slope to force logit to increase with dist."""

    def __init__(self):
        super().__init__()
        self.w_raw = nn.Parameter(torch.tensor([[0.0]]))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.w_raw)  # w >= 0
        return r * w + self.b


class NoiseMetricResNet18HF(nn.Module):
    """Shared HF ResNet-18 backbone -> global embeddings -> distance-based logits."""

    def __init__(self, model_path: Path):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_path)
        self.dist_head = PositiveDistHead()

    def encode_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,H,W)
        return: (B,C) channel-normalized global embedding
        """
        feat = self.backbone(x).last_hidden_state  # (B,C,h,w)
        feat = F.normalize(feat, p=2, dim=1)
        emb = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B,C)
        return emb

    def forward(self, face_ms: torch.Tensor, bg_ms: torch.Tensor) -> dict:
        v_face = self.encode_embed(face_ms)  # (B,C)
        v_bg = self.encode_embed(bg_ms)      # (B,C)
        diff = v_face - v_bg
        dist = diff.pow(2).sum(dim=1).sqrt()  # (B,)
        logit = self.dist_head(dist.unsqueeze(1)).squeeze(1)  # (B,)
        return {"logit": logit, "dist": dist, "feat_diff": diff}


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    cls_loss_sum = 0.0
    supcon_loss_sum = 0.0
    roi_gap_loss_sum = 0.0
    roi_rank_loss_sum = 0.0
    real_dist_sum = 0.0
    real_dist_cnt = 0
    fake_dist_sum = 0.0
    fake_dist_cnt = 0
    n_samples = 0
    correct = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        face = batch["face_ms"]
        bg = batch["bg_ms"]
        y = batch["label"]

        if not isinstance(face, torch.Tensor):
            face = torch.from_numpy(face)
            bg = torch.from_numpy(bg)
            y = torch.from_numpy(y)

        face = face.to(device).float()
        bg = bg.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad()
        out = model(face, bg)
        logit = out["logit"]
        dist = out["dist"]          # r: (B,)
        feat_diff = out["feat_diff"]

        loss_cls = bce(logit, y)
        supcon_loss = torch.tensor(0.0, device=device)
        if args.supcon_weight > 0:
            z = F.normalize(feat_diff, p=2, dim=1)
            labels = y.long()
            bs = z.size(0)
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(z.device)
            mask = mask.fill_diagonal_(0)

            sim = torch.matmul(z, z.T) / args.supcon_temperature
            logits_max, _ = sim.max(dim=1, keepdim=True)
            logits = sim - logits_max.detach()
            exp_logits = torch.exp(logits) * (1 - torch.eye(bs, device=z.device))
            denom = exp_logits.sum(dim=1, keepdim=True) + 1e-12
            num = exp_logits * mask
            pos_sum = num.sum(dim=1)
            pos_count = mask.sum(dim=1)
            valid = pos_count > 0
            if valid.any():
                log_prob = torch.log(pos_sum[valid] / denom[valid].squeeze(1))
                supcon_loss = -log_prob.mean()

        # ROI mean gap: encourage fake dist mean > real by a margin
        roi_gap_loss = torch.tensor(0.0, device=device)
        if args.roi_gap_weight > 0:
            real_mask = y == 0
            fake_mask = y == 1
            if real_mask.any() and fake_mask.any():
                mu_real = dist[real_mask].mean()
                mu_fake = dist[fake_mask].mean()
                gap = mu_fake - mu_real
                roi_gap_loss = F.relu(args.roi_gap_margin - gap).pow(2)

        # ROI ranking on uncertain samples near decision boundary
        roi_rank_loss = torch.tensor(0.0, device=device)
        if args.roi_rank_weight > 0:
            with torch.no_grad():
                probs = torch.sigmoid(logit)
                amb_mask = (probs > args.roi_rank_pmin) & (probs < args.roi_rank_pmax)

            real_amb = amb_mask & (y == 0)
            fake_amb = amb_mask & (y == 1)
            if real_amb.any() and fake_amb.any():
                r_real = dist[real_amb]
                r_fake = dist[fake_amb]
                diff = r_fake.view(-1, 1) - r_real.view(1, -1)
                viol = F.relu(args.roi_rank_margin - diff)
                roi_rank_loss = (viol.pow(2)).mean()

        loss = (
            loss_cls
            + args.supcon_weight * supcon_loss
            + args.roi_gap_weight * roi_gap_loss
            + args.roi_rank_weight * roi_rank_loss
        )

        loss.backward()
        optimizer.step()

        bs = face.size(0)
        total_loss += float(loss.item()) * bs
        cls_loss_sum += float(loss_cls.item()) * bs
        supcon_loss_sum += float(supcon_loss.item()) * bs
        roi_gap_loss_sum += float(roi_gap_loss.item()) * bs
        roi_rank_loss_sum += float(roi_rank_loss.item()) * bs
        n_samples += bs
        with torch.no_grad():
            preds = (torch.sigmoid(logit) >= 0.5).float()
            correct += (preds == y).sum().item()
            d = dist.detach()
            real_mask = y == 0
            fake_mask = y == 1
            if real_mask.any():
                real_dist_sum += d[real_mask].sum().item()
                real_dist_cnt += int(real_mask.sum().item())
            if fake_mask.any():
                fake_dist_sum += d[fake_mask].sum().item()
                fake_dist_cnt += int(fake_mask.sum().item())

        avg_total = total_loss / max(1, n_samples)
        avg_cls = cls_loss_sum / max(1, n_samples)
        avg_sup = supcon_loss_sum / max(1, n_samples)
        avg_gap = roi_gap_loss_sum / max(1, n_samples)
        avg_rank = roi_rank_loss_sum / max(1, n_samples)
        avg_acc = correct / max(1, n_samples)
        pbar.set_postfix(
            ordered_dict={
                "total": f"{avg_total:.4f}",
                "cls": f"{avg_cls:.4f}",
                "supcon": f"{avg_sup:.4f}",
                "gap": f"{avg_gap:.4f}",
                "rank": f"{avg_rank:.4f}",
                "acc": f"{avg_acc:.4f}",
            }
        )

    denom = max(1, n_samples)
    return (
        total_loss / denom,
        cls_loss_sum / denom,
        supcon_loss_sum / denom,
        roi_gap_loss_sum / denom,
        roi_rank_loss_sum / denom,
        (real_dist_sum / real_dist_cnt) if real_dist_cnt > 0 else float("nan"),
        (fake_dist_sum / fake_dist_cnt) if fake_dist_cnt > 0 else float("nan"),
    )


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        face = batch["face_ms"]
        bg = batch["bg_ms"]
        y = batch["label"]

        if not isinstance(face, torch.Tensor):
            face = torch.from_numpy(face)
            bg = torch.from_numpy(bg)
            y = torch.from_numpy(y)

        face = face.to(device).float()
        bg = bg.to(device).float()
        y = y.to(device).float()

        out = model(face, bg)
        logit = out["logit"]
        all_logits.append(logit.cpu().numpy())  # (B,)
        all_labels.append(y.cpu().numpy())

    if not all_labels:
        return {"acc": 0.0, "auc": float("nan"), "f1": 0.0, "ap": float("nan")}

    logits = np.concatenate(all_logits)  # (N,)
    labels = np.concatenate(all_labels)

    probs_class1 = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs_class1 >= 0.5).astype(int)

    labels_int = labels.astype(int)
    if np.unique(labels_int).size < 2:
        acc = accuracy_score(labels_int, preds)
        f1 = f1_score(labels_int, preds, average="macro", zero_division=0.0)
        auc = float("nan")
        ap = float("nan")
    else:
        acc = balanced_accuracy_score(labels_int, preds)
        f1 = f1_score(labels_int, preds, average="macro", zero_division=0.0)
        auc = roc_auc_score(labels_int, probs_class1)
        ap = average_precision_score(labels_int, probs_class1)

    return {"acc": acc, "auc": auc, "f1": f1, "ap": ap}


def main():
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.txt"
    ckpt_path = run_dir / "best_noise_metric_resnet18.pt"

    def write_log(msg: str):
        print(msg)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_noise_loaders(args)

    model = NoiseMetricResNet18HF(model_path=args.model_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, args)

    best_val_auc = -math.inf
    best_test_auc = -math.inf
    best_test_ckpt = None
    last_saved_ckpt = None

    def _fmt(x: float) -> str:
        if x != x:  # NaN
            return "nan"
        return f"{x:.4f}"

    for epoch in range(1, args.epochs + 1):
        (
            train_loss,
            train_cls_loss,
            train_supcon_loss,
            train_roi_gap_loss,
            train_roi_rank_loss,
            train_mu_real,
            train_mu_fake,
        ) = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics = eval_epoch(model, val_loader, device)
        test_metrics = eval_epoch(model, test_loader, device)
        scheduler.step()

        log_line = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} cls={train_cls_loss:.4f} supcon={train_supcon_loss:.4f} "
            f"gap={train_roi_gap_loss:.4f} rank={train_roi_rank_loss:.4f} | "
            f"mu_real={_fmt(train_mu_real)} mu_fake={_fmt(train_mu_fake)} | "
            f"val acc={_fmt(val_metrics['acc'])} auc={_fmt(val_metrics['auc'])} "
            f"f1={_fmt(val_metrics['f1'])} ap={_fmt(val_metrics['ap'])} | "
            f"test acc={_fmt(test_metrics['acc'])} auc={_fmt(test_metrics['auc'])} "
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
            ckpt_name = f"noise_metric_resnet18_e{epoch:03d}_test{test_metrics['auc']:.4f}.pt"
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
            if test_metrics["auc"] >= best_test_auc:
                best_test_auc = test_metrics["auc"]
                best_test_ckpt = ckpt_path_epoch
            write_log(f"  ↳ Saved checkpoint {ckpt_name} (val_auc={val_metrics['auc']:.4f}, test_auc={test_metrics['auc']:.4f})")

    # target_ckpt = best_test_ckpt or last_saved_ckpt
    # if target_ckpt is None:
    #     write_log("No checkpoint met saving criteria; skip final reload/eval.")
    # else:
    #     ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
    #     model.load_state_dict(ckpt["model_state"])
    #     test_metrics = eval_epoch(model, test_loader, device)
    #     write_log(
    #         f"[TEST] acc={_fmt(test_metrics['acc'])} auc={_fmt(test_metrics['auc'])} "
    #         f"f1={_fmt(test_metrics['f1'])} ap={_fmt(test_metrics['ap'])}"
    #     )


if __name__ == "__main__":
    main()
