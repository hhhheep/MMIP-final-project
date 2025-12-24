#!/usr/bin/env python3
"""
PyH264 Contrastive Learning Training Script
Implements Content-Quantization Invariant Learning (CQIL) using ResNet18
With Mode Collapse detection and enhanced monitoring
Following bkChen dataset standard with cross_generator mode
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU devices
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Paths
BASE_PATH = Path('/ssd5/ia313553058/MMIP')
OUTPUT_BASE = BASE_PATH / 'c23/OutPut/Offline_pre/pyh264_images_processed'
CSV_PATH = OUTPUT_BASE / 'csv'
CHECKPOINT_DIR = BASE_PATH / 'checkpoints/pyh264_contrastive'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class PyH264Dataset(Dataset):
    """Dataset for loading QP23/QP40 image pairs"""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file: Path to CSV file with columns: video_id, label, qp23_img_path, qp40_img_path, ...
            transform: Torchvision transforms to apply
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        # Verify all files exist
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            qp23_path = Path(row['qp23_img_path'])
            qp40_path = Path(row['qp40_img_path'])
            if qp23_path.exists() and qp40_path.exists():
                self.valid_indices.append(idx)

        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        logger.info(f"Loaded dataset with {len(self.df)} valid pairs from {csv_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load QP23 and QP40 images
        img_qp23 = Image.open(row['qp23_img_path']).convert('RGB')
        img_qp40 = Image.open(row['qp40_img_path']).convert('RGB')

        # Apply transforms
        if self.transform:
            img_qp23 = self.transform(img_qp23)
            img_qp40 = self.transform(img_qp40)

        return {
            'qp23': img_qp23,
            'qp40': img_qp40,
            'label': torch.tensor(row['label'], dtype=torch.float32),
            'video_id': str(row['video_id']),
            'frame_idx': row.get('frame_idx', 0)
        }


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""

    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


class ContrastiveModel(nn.Module):
    """ResNet18 with projection head for contrastive learning"""

    def __init__(self, pretrained=True):
        super().__init__()
        # Load ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        # Remove final FC layer
        self.backbone.fc = nn.Identity()
        # Add projection head
        self.projection = ProjectionHead(input_dim=512, hidden_dim=512, output_dim=128)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss with similarity tracking"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, return_similarities=False):
        """
        Args:
            z1: Embeddings for QP23 images [batch_size, dim]
            z2: Embeddings for QP40 images [batch_size, dim]
            return_similarities: If True, return (loss, pos_sim, neg_sim)

        Returns:
            loss: Contrastive loss
            pos_sim (optional): Mean positive pair similarity (raw cosine)
            neg_sim (optional): Mean negative pair similarity (raw cosine)
        """
        batch_size = z1.shape[0]
        device = z1.device

        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, dim]

        # Compute raw cosine similarity matrix (before temperature scaling)
        raw_sim = torch.mm(z, z.t())  # [2*batch_size, 2*batch_size]

        # Temperature-scaled similarity for loss computation
        sim = raw_sim / self.temperature

        # Create positive pair mask
        pos_mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Create negative pair mask (exclude diagonal and positive pairs)
        neg_mask = ~torch.eye(2*batch_size, dtype=torch.bool, device=device)
        neg_mask = neg_mask & ~pos_mask

        # Compute loss using temperature-scaled similarities
        exp_sim = torch.exp(sim)
        pos_exp = exp_sim * pos_mask
        neg_exp = exp_sim * (~torch.eye(2*batch_size, dtype=torch.bool, device=device))

        pos_sum = pos_exp.sum(dim=1)
        neg_sum = neg_exp.sum(dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        loss = loss.mean()

        if return_similarities:
            # Compute mean similarities using RAW cosine (not temperature-scaled)
            # Positive pairs: diagonal of top-right and bottom-left quadrants
            pos_similarities = raw_sim[pos_mask]
            mean_pos_sim = pos_similarities.mean().item()

            # Negative pairs: all non-diagonal, non-positive entries
            neg_similarities = raw_sim[neg_mask]
            mean_neg_sim = neg_similarities.mean().item()

            return loss, mean_pos_sim, mean_neg_sim

        return loss


def check_mode_collapse(pos_sim, neg_sim, pos_threshold=0.9, neg_threshold=0.8):
    """
    Check for mode collapse in contrastive learning.

    Mode collapse occurs when:
    - Positive pairs have very high similarity (good) BUT
    - Negative pairs ALSO have high similarity (bad - model outputs constant)

    Args:
        pos_sim: Mean positive pair similarity
        neg_sim: Mean negative pair similarity
        pos_threshold: Threshold for positive similarity
        neg_threshold: Threshold for negative similarity

    Returns:
        tuple: (is_collapsed: bool, message: str or None)
    """
    if pos_sim > pos_threshold and neg_sim > neg_threshold:
        return True, (f"MODE COLLAPSE WARNING: pos_sim={pos_sim:.4f} > {pos_threshold} "
                      f"AND neg_sim={neg_sim:.4f} > {neg_threshold}. "
                      f"Model may be outputting constant embeddings!")

    # Additional warning for concerning patterns
    if neg_sim > 0.7:
        return False, (f"CAUTION: Negative similarity is high ({neg_sim:.4f}). "
                       f"Monitor for potential collapse.")

    return False, None


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch with similarity tracking"""
    model.train()
    total_loss = 0
    total_pos_sim = 0
    total_neg_sim = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    for batch in pbar:
        # Move to device
        img_qp23 = batch['qp23'].to(device)
        img_qp40 = batch['qp40'].to(device)

        # Forward pass
        z1 = model(img_qp23)
        z2 = model(img_qp40)

        # Compute loss with similarities
        loss, pos_sim, neg_sim = criterion(z1, z2, return_similarities=True)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_pos_sim += pos_sim
        total_neg_sim += neg_sim
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pos_sim': f'{pos_sim:.3f}',
            'neg_sim': f'{neg_sim:.3f}'
        })

    avg_loss = total_loss / num_batches
    avg_pos_sim = total_pos_sim / num_batches
    avg_neg_sim = total_neg_sim / num_batches

    return avg_loss, avg_pos_sim, avg_neg_sim


def validate(model, dataloader, criterion, device):
    """Validate model with full similarity tracking"""
    model.eval()
    total_pos_sim = 0
    total_neg_sim = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move to device
            img_qp23 = batch['qp23'].to(device)
            img_qp40 = batch['qp40'].to(device)

            # Get embeddings
            z1 = model(img_qp23)
            z2 = model(img_qp40)

            # Compute similarities using criterion
            _, pos_sim, neg_sim = criterion(z1, z2, return_similarities=True)

            total_pos_sim += pos_sim
            total_neg_sim += neg_sim
            num_batches += 1

    avg_pos_sim = total_pos_sim / num_batches
    avg_neg_sim = total_neg_sim / num_batches

    return avg_pos_sim, avg_neg_sim


def main():
    """Main training function"""
    logger.info("Starting PyH264 Contrastive Learning Training")

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 4

    # Data transforms (following bkChen standard)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_csv = CSV_PATH / 'train_metadata.csv'
    val_csv = CSV_PATH / 'val_metadata.csv'

    if not train_csv.exists():
        logger.error(f"Training CSV not found: {train_csv}")
        logger.error("Please run preprocess.py first to generate the metadata CSVs.")
        return

    train_dataset = PyH264Dataset(train_csv, transform=transform)
    val_dataset = PyH264Dataset(val_csv, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

    # Create model
    model = ContrastiveModel(pretrained=True)

    # Use single GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device)

    # Loss and optimizer
    criterion = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training history (extended)
    history = {
        'train_loss': [],
        'train_pos_sim': [],
        'train_neg_sim': [],
        'val_pos_sim': [],
        'val_neg_sim': [],
        'discrimination_score': []
    }

    # Training loop
    best_discrimination_score = -float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        # Train
        train_loss, train_pos_sim, train_neg_sim = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        logger.info(f"Training - Loss: {train_loss:.4f}, "
                    f"Pos_Sim: {train_pos_sim:.4f}, Neg_Sim: {train_neg_sim:.4f}")

        # Validate
        val_pos_sim, val_neg_sim = validate(model, val_loader, criterion, device)
        logger.info(f"Validation - Pos_Sim: {val_pos_sim:.4f}, Neg_Sim: {val_neg_sim:.4f}")

        # Calculate discrimination score
        discrimination_score = val_pos_sim - val_neg_sim
        logger.info(f"Discrimination Score: {discrimination_score:.4f}")

        # Check for mode collapse
        is_collapsed, collapse_msg = check_mode_collapse(val_pos_sim, val_neg_sim)
        if collapse_msg:
            logger.warning(collapse_msg)
        if is_collapsed:
            logger.error("Training may be failing. Consider reducing learning rate "
                         "or checking data preprocessing.")

        # Update history
        history['train_loss'].append(train_loss)
        history['train_pos_sim'].append(train_pos_sim)
        history['train_neg_sim'].append(train_neg_sim)
        history['val_pos_sim'].append(val_pos_sim)
        history['val_neg_sim'].append(val_neg_sim)
        history['discrimination_score'].append(discrimination_score)

        # Save checkpoint based on discrimination score (larger gap = better discrimination)
        if discrimination_score > best_discrimination_score:
            best_discrimination_score = discrimination_score
            checkpoint_path = CHECKPOINT_DIR / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_pos_sim': val_pos_sim,
                'val_neg_sim': val_neg_sim,
                'discrimination_score': discrimination_score,
            }, checkpoint_path)
            logger.info(f"Saved best model with discrimination score: {discrimination_score:.4f}")

        # Save regular checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_pos_sim': val_pos_sim,
                'val_neg_sim': val_neg_sim,
                'discrimination_score': discrimination_score,
            }, checkpoint_path)

    # Save training history
    history_path = CHECKPOINT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"Best discrimination score: {best_discrimination_score:.4f}")
    logger.info(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()
