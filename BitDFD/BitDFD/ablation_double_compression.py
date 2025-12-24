#!/usr/bin/env python3
"""
Ablation Study: Verify "Double Compression" Hypothesis

This script tests whether the re-encoding step (C23 → PyH264) destroyed
forgery features, causing Track A to fail (AUC < 50%).

Experiment Design:
- Group A (Native): Original FF++ C23 frames
- Group B (Re-encoded): PyH264 re-encoded frames (QP23)

Both groups use a standard ImageNet-pretrained ResNet18 (NOT our failed CQIL model)
as a neutral feature extractor.

Hypothesis: If AUC(Native) >> AUC(Re-encoded), double compression destroyed features.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Paths
BASE_PATH = Path('/ssd5/ia313553058/MMIP')
NATIVE_PATH = BASE_PATH / 'c23'  # Original FF++ C23 frames
REENCODED_PATH = BASE_PATH / 'c23/OutPut/Offline_pre/pyh264_images_processed'
RESULTS_DIR = BASE_PATH / 'checkpoints/ablation_double_compression'

# Cross-generator setting (same as main experiment)
TRAIN_FAKE_METHOD = 'Deepfakes'
TEST_FAKE_METHOD = 'FaceSwap'
REAL_METHOD = 'Real_youtube'

# Sampling config (to make experiment tractable)
MAX_FRAMES_PER_VIDEO = 10  # Sample up to 10 frames per video
MAX_VIDEOS_PER_CLASS = 200  # Cap videos per class


class NativeFrameDataset(Dataset):
    """Dataset for loading original FF++ C23 frames (Native)"""

    def __init__(self, split, fake_method, real_method, transform=None,
                 max_frames=10, max_videos=200):
        self.transform = transform
        self.samples = []

        base = NATIVE_PATH / split

        # Load fake samples
        fake_dir = base / fake_method
        if fake_dir.exists():
            video_dirs = sorted([d for d in fake_dir.iterdir() if d.is_dir()])[:max_videos]
            for video_dir in video_dirs:
                frames = sorted(video_dir.glob('*.png'))
                # Filter out bbox files
                frames = [f for f in frames if 'bbox' not in f.name][:max_frames]
                for frame in frames:
                    self.samples.append({
                        'path': str(frame),
                        'label': 1,  # Fake
                        'video_id': video_dir.name,
                        'method': fake_method
                    })

        # Load real samples
        real_dir = base / real_method
        if real_dir.exists():
            video_dirs = sorted([d for d in real_dir.iterdir() if d.is_dir()])[:max_videos]
            for video_dir in video_dirs:
                frames = sorted(video_dir.glob('*.png'))
                frames = [f for f in frames if 'bbox' not in f.name][:max_frames]
                for frame in frames:
                    self.samples.append({
                        'path': str(frame),
                        'label': 0,  # Real
                        'video_id': video_dir.name,
                        'method': real_method
                    })

        logger.info(f"NativeDataset [{split}]: {len(self.samples)} samples "
                   f"(Fake={sum(1 for s in self.samples if s['label']==1)}, "
                   f"Real={sum(1 for s in self.samples if s['label']==0)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['video_id']
        }


class ReencodedFrameDataset(Dataset):
    """Dataset for loading PyH264 re-encoded frames"""

    def __init__(self, split, fake_method, real_method, transform=None,
                 max_videos=200, max_frames=10, use_qp='qp23'):
        self.transform = transform
        self.use_qp = use_qp
        self.samples = []

        base = REENCODED_PATH / split

        # Load fake samples
        fake_dir = base / fake_method
        if fake_dir.exists():
            video_dirs = sorted([d for d in fake_dir.iterdir() if d.is_dir()])[:max_videos]
            for video_dir in video_dirs:
                # Find all QP23/QP40 images in the video folder
                # Format: {frame_number}_qp23.png or {frame_number}_qp40.png
                pattern = f'*_{use_qp}.png'
                frames = sorted(video_dir.glob(pattern))[:max_frames]
                for frame in frames:
                    self.samples.append({
                        'path': str(frame),
                        'label': 1,  # Fake
                        'video_id': video_dir.name,
                        'method': fake_method
                    })

        # Load real samples
        real_dir = base / real_method
        if real_dir.exists():
            video_dirs = sorted([d for d in real_dir.iterdir() if d.is_dir()])[:max_videos]
            for video_dir in video_dirs:
                pattern = f'*_{use_qp}.png'
                frames = sorted(video_dir.glob(pattern))[:max_frames]
                for frame in frames:
                    self.samples.append({
                        'path': str(frame),
                        'label': 0,  # Real
                        'video_id': video_dir.name,
                        'method': real_method
                    })

        logger.info(f"ReencodedDataset [{split}]: {len(self.samples)} samples "
                   f"(Fake={sum(1 for s in self.samples if s['label']==1)}, "
                   f"Real={sum(1 for s in self.samples if s['label']==0)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['video_id']
        }


def get_imagenet_resnet18(device):
    """Load standard ImageNet-pretrained ResNet18 (NOT our CQIL model)"""
    logger.info("Loading ImageNet-pretrained ResNet18 (neutral feature extractor)")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # Remove classification head, output 512-dim features
    model = model.to(device)
    model.eval()
    return model


def extract_features(model, dataloader, device, desc="Extracting"):
    """Extract features from dataloader using the model"""
    features_list = []
    labels_list = []
    video_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['image'].to(device)
            labels = batch['label']

            features = model(images)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            video_ids.extend(batch['video_id'])

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    return X, y, video_ids


def evaluate_features(X_train, y_train, X_test, y_test, name):
    """Train classifier and evaluate AUC"""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    has_both_classes = len(np.unique(y_test)) > 1
    metrics = {
        'name': name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba)) if has_both_classes else 0.0,
        'ap': float(average_precision_score(y_test, y_proba)) if has_both_classes else 0.0,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_fake_ratio': float(np.mean(y_train)),
        'test_fake_ratio': float(np.mean(y_test))
    }

    return metrics


def main():
    logger.info("="*70)
    logger.info("Ablation Study: Double Compression Hypothesis")
    logger.info("="*70)
    logger.info(f"Hypothesis: Re-encoding (C23 → PyH264) destroyed forgery features")
    logger.info(f"Method: Compare AUC of Native C23 vs Re-encoded frames")
    logger.info(f"Feature Extractor: ImageNet-pretrained ResNet18 (neutral)")
    logger.info("="*70)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load neutral feature extractor (ImageNet-pretrained)
    model = get_imagenet_resnet18(device)

    # Data transforms (standard ImageNet preprocessing)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    BATCH_SIZE = 64
    NUM_WORKERS = 4

    results = {}

    # ========================================
    # Group A: Native C23 Frames
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("Group A: Native C23 Frames (Original FF++)")
    logger.info("="*70)

    # Train dataset: Deepfakes + Real_youtube
    native_train = NativeFrameDataset(
        split='train',
        fake_method=TRAIN_FAKE_METHOD,
        real_method=REAL_METHOD,
        transform=transform,
        max_frames=MAX_FRAMES_PER_VIDEO,
        max_videos=MAX_VIDEOS_PER_CLASS
    )

    # Test dataset: FaceSwap + Real_youtube (Cross-generator)
    native_test = NativeFrameDataset(
        split='test',
        fake_method=TEST_FAKE_METHOD,
        real_method=REAL_METHOD,
        transform=transform,
        max_frames=MAX_FRAMES_PER_VIDEO,
        max_videos=MAX_VIDEOS_PER_CLASS
    )

    native_train_loader = DataLoader(native_train, batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=NUM_WORKERS)
    native_test_loader = DataLoader(native_test, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=NUM_WORKERS)

    logger.info("\nExtracting Native features...")
    X_train_native, y_train_native, _ = extract_features(
        model, native_train_loader, device, "Native Train"
    )
    X_test_native, y_test_native, _ = extract_features(
        model, native_test_loader, device, "Native Test"
    )

    logger.info(f"Native Train: {X_train_native.shape}, Fake ratio: {np.mean(y_train_native):.2%}")
    logger.info(f"Native Test: {X_test_native.shape}, Fake ratio: {np.mean(y_test_native):.2%}")

    metrics_native = evaluate_features(
        X_train_native, y_train_native,
        X_test_native, y_test_native,
        "Native C23"
    )
    results['native'] = metrics_native
    logger.info(f"\nNative C23 Results:")
    logger.info(f"  ACC: {metrics_native['accuracy']:.4f}")
    logger.info(f"  AUC: {metrics_native['auc_roc']:.4f}")
    logger.info(f"  Recall: {metrics_native['recall']:.4f}")

    # ========================================
    # Group B: Re-encoded Frames (QP23)
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("Group B: Re-encoded Frames (PyH264 QP23)")
    logger.info("="*70)

    reenc_train = ReencodedFrameDataset(
        split='train',
        fake_method=TRAIN_FAKE_METHOD,
        real_method=REAL_METHOD,
        transform=transform,
        max_videos=MAX_VIDEOS_PER_CLASS,
        max_frames=MAX_FRAMES_PER_VIDEO,
        use_qp='qp23'
    )

    reenc_test = ReencodedFrameDataset(
        split='test',
        fake_method=TEST_FAKE_METHOD,
        real_method=REAL_METHOD,
        transform=transform,
        max_videos=MAX_VIDEOS_PER_CLASS,
        max_frames=MAX_FRAMES_PER_VIDEO,
        use_qp='qp23'
    )

    reenc_train_loader = DataLoader(reenc_train, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=NUM_WORKERS)
    reenc_test_loader = DataLoader(reenc_test, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)

    logger.info("\nExtracting Re-encoded features...")
    X_train_reenc, y_train_reenc, _ = extract_features(
        model, reenc_train_loader, device, "Re-encoded Train"
    )
    X_test_reenc, y_test_reenc, _ = extract_features(
        model, reenc_test_loader, device, "Re-encoded Test"
    )

    logger.info(f"Re-encoded Train: {X_train_reenc.shape}, Fake ratio: {np.mean(y_train_reenc):.2%}")
    logger.info(f"Re-encoded Test: {X_test_reenc.shape}, Fake ratio: {np.mean(y_test_reenc):.2%}")

    metrics_reenc = evaluate_features(
        X_train_reenc, y_train_reenc,
        X_test_reenc, y_test_reenc,
        "Re-encoded (QP23)"
    )
    results['reencoded_qp23'] = metrics_reenc
    logger.info(f"\nRe-encoded (QP23) Results:")
    logger.info(f"  ACC: {metrics_reenc['accuracy']:.4f}")
    logger.info(f"  AUC: {metrics_reenc['auc_roc']:.4f}")
    logger.info(f"  Recall: {metrics_reenc['recall']:.4f}")

    # ========================================
    # Group C: Re-encoded Frames (QP40) - Bonus
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("Group C: Re-encoded Frames (PyH264 QP40) - Higher Compression")
    logger.info("="*70)

    reenc_train_qp40 = ReencodedFrameDataset(
        split='train',
        fake_method=TRAIN_FAKE_METHOD,
        real_method=REAL_METHOD,
        transform=transform,
        max_videos=MAX_VIDEOS_PER_CLASS,
        max_frames=MAX_FRAMES_PER_VIDEO,
        use_qp='qp40'
    )

    reenc_test_qp40 = ReencodedFrameDataset(
        split='test',
        fake_method=TEST_FAKE_METHOD,
        real_method=REAL_METHOD,
        transform=transform,
        max_videos=MAX_VIDEOS_PER_CLASS,
        max_frames=MAX_FRAMES_PER_VIDEO,
        use_qp='qp40'
    )

    reenc_train_loader_qp40 = DataLoader(reenc_train_qp40, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=NUM_WORKERS)
    reenc_test_loader_qp40 = DataLoader(reenc_test_qp40, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS)

    logger.info("\nExtracting Re-encoded QP40 features...")
    X_train_qp40, y_train_qp40, _ = extract_features(
        model, reenc_train_loader_qp40, device, "Re-encoded QP40 Train"
    )
    X_test_qp40, y_test_qp40, _ = extract_features(
        model, reenc_test_loader_qp40, device, "Re-encoded QP40 Test"
    )

    metrics_qp40 = evaluate_features(
        X_train_qp40, y_train_qp40,
        X_test_qp40, y_test_qp40,
        "Re-encoded (QP40)"
    )
    results['reencoded_qp40'] = metrics_qp40
    logger.info(f"\nRe-encoded (QP40) Results:")
    logger.info(f"  ACC: {metrics_qp40['accuracy']:.4f}")
    logger.info(f"  AUC: {metrics_qp40['auc_roc']:.4f}")
    logger.info(f"  Recall: {metrics_qp40['recall']:.4f}")

    # ========================================
    # Comparison Summary
    # ========================================
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS: Double Compression Hypothesis")
    print("="*90)
    print(f"Feature Extractor: ImageNet-pretrained ResNet18 (neutral, NOT CQIL)")
    print(f"Cross-Generator: Train={TRAIN_FAKE_METHOD} → Test={TEST_FAKE_METHOD}")
    print("-"*90)
    print(f"{'Group':<25} {'ACC':>10} {'AUC':>10} {'AP':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*90)

    for key, m in results.items():
        print(f"{m['name']:<25} {m['accuracy']:>10.4f} {m['auc_roc']:>10.4f} {m['ap']:>10.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print("-"*90)

    # Calculate AUC difference
    auc_native = results['native']['auc_roc']
    auc_reenc = results['reencoded_qp23']['auc_roc']
    auc_qp40 = results['reencoded_qp40']['auc_roc']
    ap_native = results['native']['ap']
    ap_reenc = results['reencoded_qp23']['ap']
    ap_qp40 = results['reencoded_qp40']['ap']

    print(f"\nAUC Comparison:")
    print(f"  Native C23:         {auc_native:.4f}")
    print(f"  Re-encoded (QP23):  {auc_reenc:.4f}  (Δ = {auc_reenc - auc_native:+.4f})")
    print(f"  Re-encoded (QP40):  {auc_qp40:.4f}  (Δ = {auc_qp40 - auc_native:+.4f})")

    print(f"\nAP Comparison:")
    print(f"  Native C23:         {ap_native:.4f}")
    print(f"  Re-encoded (QP23):  {ap_reenc:.4f}  (Δ = {ap_reenc - ap_native:+.4f})")
    print(f"  Re-encoded (QP40):  {ap_qp40:.4f}  (Δ = {ap_qp40 - ap_native:+.4f})")

    print("\n" + "="*80)
    print("HYPOTHESIS VERIFICATION")
    print("="*80)

    if auc_native > auc_reenc + 0.05:  # Significant difference threshold
        print(f"✓ CONFIRMED: Double compression DESTROYED forgery features")
        print(f"  Native AUC ({auc_native:.4f}) >> Re-encoded AUC ({auc_reenc:.4f})")
        print(f"  The re-encoding step (C23 → PyH264) washed out forgery artifacts.")
        conclusion = "CONFIRMED"
    elif auc_native < auc_reenc - 0.05:
        print(f"✗ REJECTED: Re-encoding actually IMPROVED discrimination!")
        print(f"  Native AUC ({auc_native:.4f}) < Re-encoded AUC ({auc_reenc:.4f})")
        print(f"  The re-encoding step may have enhanced forgery artifacts.")
        conclusion = "REJECTED_IMPROVED"
    else:
        print(f"? INCONCLUSIVE: Both groups perform similarly")
        print(f"  Native AUC ({auc_native:.4f}) ≈ Re-encoded AUC ({auc_reenc:.4f})")
        print(f"  Double compression is NOT the primary cause of Track A failure.")
        conclusion = "INCONCLUSIVE"

    # Additional insight: Both below 50%?
    if auc_native < 0.5 and auc_reenc < 0.5:
        print(f"\n⚠ WARNING: Both Native ({auc_native:.4f}) and Re-encoded ({auc_reenc:.4f}) have AUC < 50%")
        print(f"  This suggests the cross-generator setting (Deepfakes → FaceSwap)")
        print(f"  is fundamentally challenging, regardless of compression.")

    print("="*80)

    # Save results
    output = {
        'experiment': 'Ablation Study: Double Compression Hypothesis',
        'date': datetime.now().isoformat(),
        'hypothesis': 'Re-encoding (C23 → PyH264) destroyed forgery features',
        'feature_extractor': 'ImageNet-pretrained ResNet18',
        'cross_generator': {
            'train_fake': TRAIN_FAKE_METHOD,
            'test_fake': TEST_FAKE_METHOD,
            'real': REAL_METHOD
        },
        'config': {
            'max_frames_per_video': MAX_FRAMES_PER_VIDEO,
            'max_videos_per_class': MAX_VIDEOS_PER_CLASS
        },
        'results': results,
        'comparison': {
            'native_auc': auc_native,
            'reencoded_qp23_auc': auc_reenc,
            'reencoded_qp40_auc': auc_qp40,
            'delta_qp23': auc_reenc - auc_native,
            'delta_qp40': auc_qp40 - auc_native
        },
        'conclusion': conclusion
    }

    results_path = RESULTS_DIR / 'ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    if conclusion == "CONFIRMED":
        print("1. Consider using Native C23 frames for Track A instead of re-encoded")
        print("2. Or redesign re-encoding to preserve forgery-relevant information")
        print("3. Track B (syntax features) may be more robust because it captures")
        print("   encoding behavior rather than pixel-level artifacts")
    elif conclusion == "INCONCLUSIVE":
        print("1. Double compression is NOT the main issue")
        print("2. The cross-generator setting itself is challenging")
        print("3. Consider using multi-method training instead of single-method")
        print("4. Or use domain adaptation techniques for cross-generator transfer")
    print("="*80)


if __name__ == '__main__':
    main()
