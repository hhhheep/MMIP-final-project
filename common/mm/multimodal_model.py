"""
Multimodal deepfake model: Noise + rPPG + RGB encoders with projection/fusion heads.

Design:
- Noise encoder frozen (feature extractor only).
- rPPG encoder lightly finetuned (small LR).
- RGB encoder optionally frozen (default) or partially finetuned.
- Projection heads map each modality to a shared space, L2-normalized.
- Fusion head: concat shared embeddings -> MLP -> 2-logit classifier.
- Alignment loss: strong Noise<->rPPG, weak one-way pull for RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetForImageClassification

# Local encoders
from noise.train_noise_siamese_resnet_v2 import NoiseMetricResNet18HF  # type: ignore
from rppg.train_rppg_resnet import RPPGResNet18  # type: ignore


@dataclass
class ModalFeatureDims:
    noise: int = 512
    rppg: int = 512
    rgb: int = 512
    shared: int = 256
    fusion_hidden: int = 512


class ResNetFeatWrapper(nn.Module):
    """
    Wrapper over HF ResNetForImageClassification to extract pooled features.
    Uses the penultimate feature map averaged spatially (hidden_states[-1]).
    """

    def __init__(self, model_path: Path, num_labels: int = 2):
        super().__init__()
        self.model = ResNetForImageClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x, output_hidden_states=True)
        feat = outputs.hidden_states[-1]  # (B, C, h, w)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return pooled


class MultiModalDeepfakeModel(nn.Module):
    def __init__(
        self,
        model_path_resnet: Path,
        noise_model_path: Optional[Path] = None,
        rppg_ckpt: Optional[Path] = None,
        rgb_ckpt: Optional[Path] = None,
        dims: ModalFeatureDims = ModalFeatureDims(),
        freeze_rgb: bool = True,
        freeze_noise: bool = True,
        finetune_rppg: bool = True,
    ):
        super().__init__()
        self.freeze_noise = freeze_noise
        # Noise encoder (frozen)
        self.noise_encoder = NoiseMetricResNet18HF(model_path=model_path_resnet)
        if noise_model_path and noise_model_path.exists():
            state = torch.load(noise_model_path, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            self.noise_encoder.load_state_dict(state, strict=False)
        if freeze_noise:
            for p in self.noise_encoder.parameters():
                p.requires_grad = False

        # rPPG encoder (HF ResNet18 based)
        self.rppg_encoder = RPPGResNet18(model_path=model_path_resnet)
        if rppg_ckpt and rppg_ckpt.exists():
            state = torch.load(rppg_ckpt, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            try:
                self.rppg_encoder.load_state_dict(state, strict=False)
            except Exception:
                pass
        if not finetune_rppg:
            for p in self.rppg_encoder.parameters():
                p.requires_grad = False

        # RGB encoder (HF ResNet18 features)
        self.rgb_encoder = ResNetFeatWrapper(model_path_resnet)
        if rgb_ckpt and rgb_ckpt.exists():
            state = torch.load(rgb_ckpt, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            try:
                self.rgb_encoder.load_state_dict(state, strict=False)
            except Exception:
                pass
        if freeze_rgb:
            for p in self.rgb_encoder.parameters():
                p.requires_grad = False

        # Projection heads to shared space
        self.proj_noise = nn.Linear(dims.noise, dims.shared)
        self.proj_rppg = nn.Linear(dims.rppg, dims.shared)
        self.proj_rgb = nn.Linear(dims.rgb, dims.shared)

        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(3 * dims.shared, dims.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(dims.fusion_hidden, 2),
        )

        self.dims = dims

    # ---------- feature helpers ----------
    def _noise_video_feat(self, face_ms: torch.Tensor, bg_ms: torch.Tensor) -> torch.Tensor:
        """
        face_ms/bg_ms: shape (B, T, 3, H, W) or (B, 3, H, W)
        Returns video-level pooled embedding (B, D_noise)
        """
        if face_ms.dim() == 5:
            B, T = face_ms.shape[:2]
            face_ms_flat = face_ms.view(B * T, *face_ms.shape[2:])
            bg_ms_flat = bg_ms.view(B * T, *bg_ms.shape[2:])
        else:
            B, T = face_ms.shape[0], 1
            face_ms_flat = face_ms
            bg_ms_flat = bg_ms

        with torch.set_grad_enabled(not self.freeze_noise):
            emb_face = self.noise_encoder.encode_embed(face_ms_flat)  # (B*T, C)
            emb_bg = self.noise_encoder.encode_embed(bg_ms_flat)
            diff = emb_face - emb_bg
        diff = diff.view(B, T, -1).mean(dim=1)
        return diff

    def _rppg_video_feat(self, rppg: torch.Tensor) -> torch.Tensor:
        """
        rppg: (B, 1, H, W) -> duplicate to 3-channel inside RPPG encoder.
        Use classifier logits as feature? We instead use pre-head pooled feature.
        """
        x = rppg.repeat(1, 3, 1, 1)
        feats = self.rppg_encoder.net(x, output_hidden_states=True)
        feat_map = feats.hidden_states[-1]  # (B, C, h, w)
        pooled = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        return pooled

    def _rgb_video_feat(self, rgb_bag: torch.Tensor) -> torch.Tensor:
        """
        rgb_bag: (B, T, 3, H, W)
        Mean-pool encoder features across frames.
        """
        B, T = rgb_bag.shape[:2]
        x = rgb_bag.view(B * T, *rgb_bag.shape[2:])
        feat = self.rgb_encoder(x)  # (B*T, C)
        feat = feat.view(B, T, -1).mean(dim=1)
        return feat

    # ---------- forward ----------
    def forward(
        self,
        noise_face_ms: torch.Tensor,
        noise_bg_ms: torch.Tensor,
        rppg: torch.Tensor,
        rgb_bag: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_noise = self._noise_video_feat(noise_face_ms, noise_bg_ms)
        h_rppg = self._rppg_video_feat(rppg)
        h_rgb = self._rgb_video_feat(rgb_bag)

        # Dim sanity check to catch backbone changes early.
        assert h_noise.shape[-1] == self.dims.noise, f"Noise feat dim {h_noise.shape[-1]} != {self.dims.noise}"
        assert h_rppg.shape[-1] == self.dims.rppg, f"rPPG feat dim {h_rppg.shape[-1]} != {self.dims.rppg}"
        assert h_rgb.shape[-1] == self.dims.rgb, f"RGB feat dim {h_rgb.shape[-1]} != {self.dims.rgb}"

        s_noise = F.normalize(self.proj_noise(h_noise), dim=-1)
        s_rppg = F.normalize(self.proj_rppg(h_rppg), dim=-1)
        s_rgb = F.normalize(self.proj_rgb(h_rgb), dim=-1)

        fused = torch.cat([s_noise, s_rppg, s_rgb], dim=-1)
        logits = self.fusion_head(fused)

        return {
            "logits": logits,
            "s_noise": s_noise,
            "s_rppg": s_rppg,
            "s_rgb": s_rgb,
            "h_noise": h_noise,
            "h_rppg": h_rppg,
            "h_rgb": h_rgb,
        }


def alignment_loss(s_noise: torch.Tensor, s_rppg: torch.Tensor, s_rgb: torch.Tensor, lambda_r: float = 0.1) -> torch.Tensor:
    """
    Strong align between Noise and rPPG; weak one-way pull for RGB (teacher detached).
    """
    # Noise <-> rPPG:雙向對齊，兩邊都收梯度
    cos_np = (s_noise * s_rppg).sum(dim=-1)
    l_align_np = ((1.0 - cos_np) ** 2).mean()

    # Noise / rPPG 作為 teacher，RGB 作為 student（只有 RGB 收梯度）
    cos_nr = (s_noise.detach() * s_rgb).sum(dim=-1)
    cos_pr = (s_rppg.detach() * s_rgb).sum(dim=-1)
    l_align_nr = ((1.0 - cos_nr) ** 2).mean()
    l_align_pr = ((1.0 - cos_pr) ** 2).mean()
    return l_align_np + lambda_r * (l_align_nr + l_align_pr)


def multimodal_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, lambda_align: float = 0.2, lambda_r: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = outputs["logits"]
    s_noise = outputs["s_noise"]
    s_rppg = outputs["s_rppg"]
    s_rgb = outputs["s_rgb"]

    l_cls = F.cross_entropy(logits, labels)
    l_align = alignment_loss(s_noise, s_rppg, s_rgb, lambda_r=lambda_r)
    l_total = l_cls + lambda_align * l_align
    return l_total, {"cls": float(l_cls.item()), "align": float(l_align.item()), "total": float(l_total.item())}
