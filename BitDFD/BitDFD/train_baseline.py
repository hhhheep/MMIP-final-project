#!/usr/bin/env python3
"""
Baseline Training Script using H.264 Encoding Statistics
Trains RandomForest classifier on encoding features for deepfake detection
Following bkChen dataset standard with cross_generator mode
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
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

# Paths
BASE_PATH = Path('/ssd5/ia313553058/MMIP')
OUTPUT_BASE = BASE_PATH / 'c23/OutPut/Offline_pre/pyh264_images_processed'
CSV_PATH = OUTPUT_BASE / 'csv'
CHECKPOINT_DIR = BASE_PATH / 'checkpoints/pyh264_baseline'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_stats_from_json(stats_path: str) -> dict:
    """Load encoding statistics from JSON file."""
    try:
        with open(stats_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load stats from {stats_path}: {e}")
        return None


def extract_features_from_stats(stats_qp23: dict, stats_qp40: dict) -> np.ndarray:
    """
    Extract feature vector from encoding statistics.

    Features extracted:
    1. QP23 features (6 features)
    2. QP40 features (6 features)
    3. Difference features (6 features)

    Total: 18 core features (+ optional deep inspection features)
    """
    features = []

    # Helper function to safely get values
    def safe_get(d, key, default=0.0):
        val = d.get(key, default)
        return float(val) if val is not None else default

    # Core features for each QP level
    for stats, prefix in [(stats_qp23, 'qp23'), (stats_qp40, 'qp40')]:
        # Basic compression features
        features.append(safe_get(stats, 'bpp'))
        features.append(safe_get(stats, 'compression_ratio'))
        features.append(safe_get(stats, 'encoding_time'))

        # Coefficient statistics (may be None if deep inspection failed)
        features.append(safe_get(stats, 'non_zero_ratio'))
        features.append(safe_get(stats, 'dc_mean'))
        features.append(safe_get(stats, 'dc_std'))

    # Difference features (QP23 - QP40)
    # These capture how the image responds to different compression levels
    bpp_diff = safe_get(stats_qp23, 'bpp') - safe_get(stats_qp40, 'bpp')
    cr_diff = safe_get(stats_qp23, 'compression_ratio') - safe_get(stats_qp40, 'compression_ratio')
    time_diff = safe_get(stats_qp23, 'encoding_time') - safe_get(stats_qp40, 'encoding_time')

    nz_ratio_diff = safe_get(stats_qp23, 'non_zero_ratio') - safe_get(stats_qp40, 'non_zero_ratio')
    dc_mean_diff = safe_get(stats_qp23, 'dc_mean') - safe_get(stats_qp40, 'dc_mean')
    dc_std_diff = safe_get(stats_qp23, 'dc_std') - safe_get(stats_qp40, 'dc_std')

    features.extend([
        bpp_diff, cr_diff, time_diff,
        nz_ratio_diff, dc_mean_diff, dc_std_diff
    ])

    # Ratio features (can reveal different compression behaviors)
    bpp_ratio = safe_get(stats_qp40, 'bpp') / max(safe_get(stats_qp23, 'bpp'), 1e-6)
    cr_ratio = safe_get(stats_qp40, 'compression_ratio') / max(safe_get(stats_qp23, 'compression_ratio'), 1e-6)

    features.extend([bpp_ratio, cr_ratio])

    return np.array(features)


def load_dataset(csv_path: Path, split_name: str) -> tuple:
    """
    Load dataset and extract features from JSON stats files.

    Args:
        csv_path: Path to CSV directory
        split_name: 'train', 'val', or 'test'

    Returns:
        X: Feature matrix [N, num_features]
        y: Labels [N]
        metadata: List of dicts with video_id, method info
    """
    csv_file = csv_path / f'{split_name}_metadata.csv'

    if not csv_file.exists():
        logger.error(f"CSV file not found: {csv_file}")
        logger.error("Please run preprocess.py first to generate the metadata CSVs.")
        return None, None, None

    df = pd.read_csv(csv_file)

    logger.info(f"Loading {split_name} data from {csv_file}")
    logger.info(f"Total entries in CSV: {len(df)}")

    features_list = []
    labels_list = []
    metadata_list = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Loading {split_name}'):
        qp23_json_path = row['qp23_json_path']
        qp40_json_path = row['qp40_json_path']

        # Load stats
        stats_qp23 = load_stats_from_json(qp23_json_path)
        stats_qp40 = load_stats_from_json(qp40_json_path)

        if stats_qp23 is None or stats_qp40 is None:
            skipped += 1
            continue

        # Extract features
        try:
            features = extract_features_from_stats(stats_qp23, stats_qp40)
            features_list.append(features)
            labels_list.append(int(row['label']))
            metadata_list.append({
                'video_id': str(row['video_id']),
                'method': row.get('method', 'unknown')
            })
        except Exception as e:
            logger.warning(f"Feature extraction failed for {row['video_id']}: {e}")
            skipped += 1

    logger.info(f"Loaded {len(features_list)} samples, skipped {skipped}")

    if len(features_list) == 0:
        return None, None, None

    X = np.array(features_list)
    y = np.array(labels_list)

    return X, y, metadata_list


def get_feature_names() -> list:
    """Return list of feature names for interpretation."""
    return [
        # QP23 features
        'qp23_bpp', 'qp23_compression_ratio', 'qp23_encoding_time',
        'qp23_non_zero_ratio', 'qp23_dc_mean', 'qp23_dc_std',
        # QP40 features
        'qp40_bpp', 'qp40_compression_ratio', 'qp40_encoding_time',
        'qp40_non_zero_ratio', 'qp40_dc_mean', 'qp40_dc_std',
        # Difference features
        'bpp_diff', 'cr_diff', 'time_diff',
        'nz_ratio_diff', 'dc_mean_diff', 'dc_std_diff',
        # Ratio features
        'bpp_ratio', 'cr_ratio'
    ]


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> tuple:
    """Train Random Forest classifier."""
    logger.info("Training Random Forest classifier...")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    clf.fit(X_train_scaled, y_train)

    # Evaluate on validation
    y_pred = clf.predict(X_val_scaled)
    y_proba = clf.predict_proba(X_val_scaled)[:, 1]

    metrics = evaluate_classifier(y_val, y_pred, y_proba)

    return clf, scaler, metrics


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    # AUC-ROC and AP only if both classes present
    if len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        metrics['ap'] = average_precision_score(y_true, y_proba)
    else:
        metrics['auc_roc'] = 0.0
        metrics['ap'] = 0.0

    return metrics


def evaluate_on_test(clf, scaler, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate classifier on test set."""
    logger.info(f"\nEvaluating {model_name} on FaceSwap test set (Cross-Generator)...")

    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_proba)

    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Test Results (Cross-Generator: Train=Deepfakes, Test=FaceSwap)")
    logger.info(f"{'='*60}")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"  AP:        {metrics['ap']:.4f}")

    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Real', 'Fake'])}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


def print_feature_importance(clf, feature_names: list, top_k: int = 10):
    """Print feature importance analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 60)

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    logger.info(f"\nTop {top_k} Most Important Features:")
    for i, idx in enumerate(indices[:top_k]):
        logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Check if key features are important
    key_features = ['bpp_diff', 'cr_diff', 'time_diff']
    logger.info(f"\nKey Feature Importance (Hypothesis Verification):")
    for feat in key_features:
        if feat in feature_names:
            feat_idx = feature_names.index(feat)
            rank = list(indices).index(feat_idx) + 1
            logger.info(f"  {feat}: importance={importances[feat_idx]:.4f}, rank={rank}/{len(feature_names)}")

    return dict(zip(feature_names, importances.tolist()))


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("PyH264 Baseline Training - Encoding Statistics Classifier")
    logger.info("=" * 60)

    # Load datasets
    logger.info("\nLoading datasets...")
    X_train, y_train, meta_train = load_dataset(CSV_PATH, 'train')
    X_val, y_val, meta_val = load_dataset(CSV_PATH, 'val')
    X_test, y_test, meta_test = load_dataset(CSV_PATH, 'test')

    if X_train is None or X_val is None or X_test is None:
        logger.error("Failed to load datasets. Please run preprocess.py first.")
        return

    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train: {len(X_train)} samples, {X_train.shape[1]} features")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")

    logger.info(f"\nLabel distribution:")
    logger.info(f"  Train: Real={np.sum(y_train==0)}, Fake={np.sum(y_train==1)}")
    logger.info(f"  Val:   Real={np.sum(y_val==0)}, Fake={np.sum(y_val==1)}")
    logger.info(f"  Test:  Real={np.sum(y_test==0)}, Fake={np.sum(y_test==1)}")

    # Get feature names
    feature_names = get_feature_names()

    # Train Random Forest
    rf_clf, rf_scaler, rf_val_metrics = train_random_forest(X_train, y_train, X_val, y_val)

    logger.info(f"\nRandom Forest Validation Metrics:")
    for k, v in rf_val_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Evaluate on test set (Cross-Generator)
    rf_test_metrics = evaluate_on_test(rf_clf, rf_scaler, X_test, y_test, "Random Forest")

    # Feature importance analysis
    feature_importance = print_feature_importance(rf_clf, feature_names)

    # Save results
    results = {
        'model': 'RandomForest',
        'validation': rf_val_metrics,
        'test': rf_test_metrics,
        'feature_importance': feature_importance,
        'dataset_info': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'num_features': X_train.shape[1],
            'cross_generator': True,
            'train_fake_method': 'Deepfakes',
            'test_fake_method': 'FaceSwap'
        }
    }

    results_path = CHECKPOINT_DIR / 'baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {results_path}")

    # Save model and scaler
    model_path = CHECKPOINT_DIR / 'random_forest_model.joblib'
    scaler_path = CHECKPOINT_DIR / 'random_forest_scaler.joblib'
    joblib.dump(rf_clf, model_path)
    joblib.dump(rf_scaler, scaler_path)
    logger.info(f"Saved model to {model_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {rf_test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {rf_test_metrics['f1']:.4f}")
    logger.info(f"\nThis baseline tests if H.264 compression artifacts alone can detect deepfakes.")
    logger.info(f"Cross-Generator evaluation: Trained on Deepfakes, tested on FaceSwap.")


if __name__ == '__main__':
    main()
