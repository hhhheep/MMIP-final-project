#!/usr/bin/env python3
"""
Non-Linear Classifier Evaluation for Track A Contrastive Features

This script evaluates whether non-linear classifiers can improve upon the
Linear Probe results (AUC ~46.66%) by better handling the complex 512-dim
feature space learned by contrastive learning.

Classifiers tested:
1. Linear Probe (Logistic Regression) - Baseline
2. SVM with RBF kernel
3. MLP (Multi-Layer Perceptron)
4. Random Forest (for comparison with Track B)

All classifiers use class_weight='balanced' to handle imbalanced data.
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
from datetime import datetime

# Scikit-learn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
import joblib
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
OUTPUT_BASE = BASE_PATH / 'c23/OutPut/Offline_pre/pyh264_images_processed'
CSV_PATH = OUTPUT_BASE / 'csv'
CHECKPOINT_DIR = BASE_PATH / 'checkpoints/pyh264_contrastive'
RESULTS_DIR = BASE_PATH / 'checkpoints/pyh264_contrastive/nonlinear_eval'


class FeatureDataset(Dataset):
    """Dataset for loading images for feature extraction"""

    def __init__(self, csv_file, transform=None, use_qp='qp23'):
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

    model = ContrastiveBackbone()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract backbone weights from full model
    state_dict = checkpoint['model_state_dict']
    backbone_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            backbone_state_dict[key] = value

    model.load_state_dict(backbone_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
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

            features = model(images)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

    return X, y


def evaluate_classifier(clf, scaler, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Evaluate a classifier and return metrics."""
    X_test_scaled = scaler.transform(X_test)

    y_pred = clf.predict(X_test_scaled)

    # Get probabilities for AUC
    if hasattr(clf, 'predict_proba'):
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        # For SVM without probability, use decision_function
        y_proba = clf.decision_function(X_test_scaled)

    metrics = {
        'name': name,
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

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Per-class metrics
    metrics['true_negative'] = int(cm[0, 0])
    metrics['false_positive'] = int(cm[0, 1])
    metrics['false_negative'] = int(cm[1, 0])
    metrics['true_positive'] = int(cm[1, 1])

    return metrics


def train_and_evaluate_classifiers(X_train, y_train, X_test, y_test):
    """Train multiple classifiers and compare their performance."""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    results = []
    classifiers = {}

    # 1. Linear Probe (Logistic Regression) - Baseline
    logger.info("\n" + "="*60)
    logger.info("Training Classifier 1/4: Linear Probe (Logistic Regression)")
    logger.info("="*60)

    clf_linear = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        solver='lbfgs'
    )
    clf_linear.fit(X_train_scaled, y_train)
    metrics_linear = evaluate_classifier(clf_linear, scaler, X_test, y_test, "Linear Probe")
    results.append(metrics_linear)
    classifiers['linear'] = clf_linear
    logger.info(f"  ACC: {metrics_linear['accuracy']:.4f}, AUC: {metrics_linear['auc_roc']:.4f}")

    # 2. SVM with RBF kernel
    logger.info("\n" + "="*60)
    logger.info("Training Classifier 2/4: SVM (RBF Kernel)")
    logger.info("="*60)

    clf_svm = SVC(
        kernel='rbf',
        probability=True,  # Required for predict_proba
        class_weight='balanced',
        random_state=42,
        C=1.0,
        gamma='scale'
    )
    clf_svm.fit(X_train_scaled, y_train)
    metrics_svm = evaluate_classifier(clf_svm, scaler, X_test, y_test, "SVM (RBF)")
    results.append(metrics_svm)
    classifiers['svm'] = clf_svm
    logger.info(f"  ACC: {metrics_svm['accuracy']:.4f}, AUC: {metrics_svm['auc_roc']:.4f}")

    # 3. MLP (Multi-Layer Perceptron)
    logger.info("\n" + "="*60)
    logger.info("Training Classifier 3/4: MLP (256, 128)")
    logger.info("="*60)

    # Calculate class weights for sample_weight in MLP
    # sklearn MLPClassifier doesn't support class_weight directly
    # We use early_stopping and balanced initial weights through data

    # Compute sample weights
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    sample_weights = class_weights[y_train]

    clf_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64
    )
    # Note: MLPClassifier doesn't support sample_weight in fit()
    # We'll train without it but can try oversampling if needed
    clf_mlp.fit(X_train_scaled, y_train)
    metrics_mlp = evaluate_classifier(clf_mlp, scaler, X_test, y_test, "MLP (256,128)")
    results.append(metrics_mlp)
    classifiers['mlp'] = clf_mlp
    logger.info(f"  ACC: {metrics_mlp['accuracy']:.4f}, AUC: {metrics_mlp['auc_roc']:.4f}")

    # 4. Random Forest (for comparison with Track B)
    logger.info("\n" + "="*60)
    logger.info("Training Classifier 4/4: Random Forest")
    logger.info("="*60)

    clf_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        max_depth=20
    )
    clf_rf.fit(X_train_scaled, y_train)
    metrics_rf = evaluate_classifier(clf_rf, scaler, X_test, y_test, "Random Forest")
    results.append(metrics_rf)
    classifiers['random_forest'] = clf_rf
    logger.info(f"  ACC: {metrics_rf['accuracy']:.4f}, AUC: {metrics_rf['auc_roc']:.4f}")

    # 5. Bonus: MLP with oversampling to handle class imbalance
    logger.info("\n" + "="*60)
    logger.info("Training Classifier 5/4 (Bonus): MLP with Oversampling")
    logger.info("="*60)

    # Simple oversampling: repeat minority class samples
    minority_mask = y_train == 1
    majority_mask = y_train == 0

    X_minority = X_train_scaled[minority_mask]
    y_minority = y_train[minority_mask]
    X_majority = X_train_scaled[majority_mask]
    y_majority = y_train[majority_mask]

    # Calculate oversample factor
    oversample_factor = len(y_majority) // len(y_minority)

    # Oversample minority class
    X_minority_oversampled = np.tile(X_minority, (oversample_factor, 1))
    y_minority_oversampled = np.tile(y_minority, oversample_factor)

    # Combine
    X_train_balanced = np.vstack([X_majority, X_minority_oversampled])
    y_train_balanced = np.concatenate([y_majority, y_minority_oversampled])

    # Shuffle
    shuffle_idx = np.random.permutation(len(y_train_balanced))
    X_train_balanced = X_train_balanced[shuffle_idx]
    y_train_balanced = y_train_balanced[shuffle_idx]

    logger.info(f"  Oversampled: {len(y_minority)} -> {len(y_minority_oversampled)} minority samples")

    clf_mlp_os = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        batch_size=64
    )
    clf_mlp_os.fit(X_train_balanced, y_train_balanced)
    metrics_mlp_os = evaluate_classifier(clf_mlp_os, scaler, X_test, y_test, "MLP + Oversample")
    results.append(metrics_mlp_os)
    classifiers['mlp_oversample'] = clf_mlp_os
    logger.info(f"  ACC: {metrics_mlp_os['accuracy']:.4f}, AUC: {metrics_mlp_os['auc_roc']:.4f}")

    return results, classifiers, scaler


def print_comparison_table(results):
    """Print a formatted comparison table of all classifiers."""

    print("\n" + "="*100)
    print("CLASSIFIER COMPARISON TABLE - Track A Contrastive Features (512-dim)")
    print("="*100)
    print(f"{'Classifier':<20} {'ACC':>10} {'AUC':>10} {'AP':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*100)

    for r in results:
        print(f"{r['name']:<20} {r['accuracy']:>10.4f} {r['auc_roc']:>10.4f} {r['ap']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

    print("-"*100)

    # Find best AUC and AP
    best_auc = max(results, key=lambda x: x['auc_roc'])
    best_ap = max(results, key=lambda x: x['ap'])
    baseline_auc = results[0]['auc_roc']  # Linear Probe
    baseline_ap = results[0]['ap']

    print(f"\nBest AUC: {best_auc['name']} ({best_auc['auc_roc']:.4f})")
    print(f"Best AP:  {best_ap['name']} ({best_ap['ap']:.4f})")
    print(f"Improvement over Linear Probe: AUC {best_auc['auc_roc'] - baseline_auc:+.4f}, AP {best_ap['ap'] - baseline_ap:+.4f}")

    # Check if any classifier exceeds 50% AUC
    above_random = [r for r in results if r['auc_roc'] > 0.5]
    if above_random:
        print(f"\nClassifiers with AUC > 50% (better than random):")
        for r in above_random:
            print(f"  - {r['name']}: {r['auc_roc']:.4f}")
    else:
        print(f"\nNo classifier achieved AUC > 50%")
        print("This suggests the contrastive features may not contain discriminative information")
        print("for cross-generator detection (Deepfakes -> FaceSwap)")

    print("="*90)


def print_confusion_matrices(results):
    """Print confusion matrices for all classifiers."""
    print("\n" + "="*60)
    print("CONFUSION MATRICES")
    print("="*60)

    for r in results:
        cm = np.array(r['confusion_matrix'])
        print(f"\n{r['name']}:")
        print(f"                 Predicted")
        print(f"              Real    Fake")
        print(f"Actual Real   {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"Actual Fake   {cm[1,0]:4d}    {cm[1,1]:4d}")
        print(f"  -> Fake Recall: {r['recall']:.2%} ({r['true_positive']}/{r['true_positive']+r['false_negative']})")


def main():
    logger.info("="*60)
    logger.info("Track A Non-Linear Classifier Evaluation")
    logger.info("Testing if non-linear classifiers can improve AUC from 46.66%")
    logger.info("="*60)

    # Check for trained model
    model_path = CHECKPOINT_DIR / 'best_model.pth'
    if not model_path.exists():
        logger.error(f"Trained model not found: {model_path}")
        return

    # Check for CSVs
    train_csv = CSV_PATH / 'train_metadata.csv'
    test_csv = CSV_PATH / 'test_metadata.csv'

    if not train_csv.exists() or not test_csv.exists():
        logger.error("CSV files not found. Please run build_index.py first.")
        return

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    model = load_trained_model(model_path, device)

    # Create datasets
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
    logger.info(f"Train labels: Real={np.sum(y_train==0)}, Fake={np.sum(y_train==1)}")

    logger.info("\nExtracting test features...")
    X_test, y_test = extract_features(model, test_loader, device)
    logger.info(f"Test features shape: {X_test.shape}")
    logger.info(f"Test labels: Real={np.sum(y_test==0)}, Fake={np.sum(y_test==1)}")

    # Train and evaluate all classifiers
    results, classifiers, scaler = train_and_evaluate_classifiers(
        X_train, y_train, X_test, y_test
    )

    # Print comparison table
    print_comparison_table(results)

    # Print confusion matrices
    print_confusion_matrices(results)

    # Save results
    output = {
        'experiment': 'Track A Non-Linear Classifier Evaluation',
        'date': datetime.now().isoformat(),
        'feature_dim': 512,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_distribution': {
            'real': int(np.sum(y_train == 0)),
            'fake': int(np.sum(y_train == 1))
        },
        'test_distribution': {
            'real': int(np.sum(y_test == 0)),
            'fake': int(np.sum(y_test == 1))
        },
        'results': results,
        'best_classifier': max(results, key=lambda x: x['auc_roc'])['name'],
        'best_auc': max(results, key=lambda x: x['auc_roc'])['auc_roc']
    }

    results_path = RESULTS_DIR / 'nonlinear_comparison.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save best model
    best_name = output['best_classifier'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
    best_clf = classifiers[list(classifiers.keys())[results.index(max(results, key=lambda x: x['auc_roc']))]]

    model_save_path = RESULTS_DIR / f'best_classifier_{best_name}.joblib'
    joblib.dump(best_clf, model_save_path)

    scaler_path = RESULTS_DIR / 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_path)

    logger.info(f"Best classifier saved to {model_save_path}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    baseline_auc = results[0]['auc_roc']
    best_result = max(results, key=lambda x: x['auc_roc'])

    print(f"Original Linear Probe AUC: {baseline_auc:.4f}")
    print(f"Best Non-Linear AUC:       {best_result['auc_roc']:.4f} ({best_result['name']})")
    print(f"Improvement:               {best_result['auc_roc'] - baseline_auc:+.4f}")

    if best_result['auc_roc'] > 0.5:
        print("\nConclusion: Non-linear classifier improved AUC above random chance!")
    else:
        print("\nConclusion: Even non-linear classifiers cannot salvage the features.")
        print("The contrastive learning objective (CQIL) may need to be redesigned")
        print("for cross-generator detection.")

    print("="*60)


if __name__ == '__main__':
    main()
