#!/usr/bin/env python3
"""
Contrastive Model Evaluation Script
Evaluates the trained contrastive model using a Linear Probe (Logistic Regression).
Following bkChen dataset standard with cross_generator mode.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set GPU
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Paths
BASE_PATH = Path('/ssd5/ia313553058/MMIP')
OUTPUT_BASE = BASE_PATH / 'c23/OutPut/Offline_pre/pyh264_images_processed'
CSV_PATH = OUTPUT_BASE / 'csv'
CHECKPOINT_DIR = BASE_PATH / 'checkpoints/pyh264_contrastive'


class FeatureDataset(Dataset):
    """Dataset for loading images for feature extraction"""

    def __init__(self, csv_file, transform=None, use_qp='qp23'):
        """
        Args:
            csv_file: Path to CSV file
            transform: Torchvision transforms
            use_qp: Which QP level to use ('qp23' or 'qp40')
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_qp = use_qp

        # Verify files exist
        self.valid_indices = []
        img_col = f'{use_qp}_img_path'
        for idx, row in self.df.iterrows():
            img_path = Path(row[img_col])
            if img_path.exists():
                self.valid_indices.append(idx)

        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        logger.info(f"Loaded {len(self.df)} valid samples from {csv_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_col = f'{self.use_qp}_img_path'

        img = Image.open(row[img_col]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'video_id': str(row['video_id'])
        }


class ContrastiveBackbone(nn.Module):
    """ResNet18 backbone for feature extraction (no projection head)"""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


def load_trained_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load the trained contrastive model and return backbone only."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Create model
    model = ContrastiveBackbone()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract backbone weights from full model
    state_dict = checkpoint['model_state_dict']
    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            backbone_state_dict[key] = value

    # Load weights
    model.load_state_dict(backbone_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Best discrimination score: {checkpoint.get('discrimination_score', 'N/A')}")

    return model


def extract_features(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple:
    """Extract features from all samples in dataloader."""
    features_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch['image'].to(device)
            labels = batch['label']

            # Extract 512-dim features
            features = model(images)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    return X, y


def train_linear_probe(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Train a Logistic Regression classifier on extracted features."""
    logger.info("Training Linear Probe (Logistic Regression)...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    return clf, scaler


def evaluate(clf, scaler, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the linear probe on test set."""
    X_test_scaled = scaler.transform(X_test)

    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if len(np.unique(y_test)) > 1:
        metrics['auc_roc'] = float(roc_auc_score(y_test, y_proba))
        metrics['ap'] = float(average_precision_score(y_test, y_proba))
    else:
        metrics['auc_roc'] = 0.0
        metrics['ap'] = 0.0

    return metrics, y_pred, y_test


def main():
    logger.info("=" * 60)
    logger.info("PyH264 Contrastive Model Evaluation - Linear Probe")
    logger.info("=" * 60)

    # Check for trained model
    model_path = CHECKPOINT_DIR / 'best_model.pth'
    if not model_path.exists():
        logger.error(f"Trained model not found: {model_path}")
        logger.error("Please run train.py first.")
        return

    # Check for CSVs
    train_csv = CSV_PATH / 'train_metadata.csv'
    test_csv = CSV_PATH / 'test_metadata.csv'

    if not train_csv.exists() or not test_csv.exists():
        logger.error("CSV files not found. Please run build_index.py first.")
        return

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    model = load_trained_model(model_path, device)

    # Create datasets (use QP23 for evaluation - high quality version)
    train_dataset = FeatureDataset(train_csv, transform=transform, use_qp='qp23')
    test_dataset = FeatureDataset(test_csv, transform=transform, use_qp='qp23')

    BATCH_SIZE = 64
    NUM_WORKERS = 4

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # Extract features
    logger.info("\nExtracting training features...")
    X_train, y_train = extract_features(model, train_loader, device)
    logger.info(f"Train features shape: {X_train.shape}")

    logger.info("\nExtracting test features...")
    X_test, y_test = extract_features(model, test_loader, device)
    logger.info(f"Test features shape: {X_test.shape}")

    # Train linear probe
    clf, scaler = train_linear_probe(X_train, y_train)

    # Evaluate
    metrics, y_pred, y_true = evaluate(clf, scaler, X_test, y_test)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Contrastive Linear Probe Results (Cross-Generator)")
    logger.info("Train: Deepfakes -> Test: FaceSwap")
    logger.info("=" * 60)
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"  AP:        {metrics['ap']:.4f}")

    logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=['Real', 'Fake'])}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    # Save results
    results = {
        'model': 'Contrastive Linear Probe',
        'backbone': 'ResNet18',
        'feature_dim': 512,
        'test_metrics': metrics,
        'dataset_info': {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'cross_generator': True,
            'train_fake_method': 'Deepfakes',
            'test_fake_method': 'FaceSwap'
        }
    }

    results_path = CHECKPOINT_DIR / 'linear_probe_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save linear probe model
    probe_path = CHECKPOINT_DIR / 'linear_probe_model.joblib'
    scaler_path = CHECKPOINT_DIR / 'linear_probe_scaler.joblib'
    joblib.dump(clf, probe_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Linear probe saved to {probe_path}")

    # Final summary for easy copy-paste
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS (Copy to After_preprocess.md)")
    logger.info("=" * 60)
    logger.info(f"ACC_A = {metrics['accuracy']:.4f}")
    logger.info(f"AUC_A = {metrics['auc_roc']:.4f}")
    logger.info(f"AP_A  = {metrics['ap']:.4f}")


if __name__ == '__main__':
    main()
