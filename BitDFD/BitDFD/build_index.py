#!/usr/bin/env python3
"""
Build Index Script - Generate Master CSVs from processed files
Scans the output directory and pairs QP23/QP40 files to create training CSVs.
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging
from tqdm import tqdm

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
CSV_PATH.mkdir(parents=True, exist_ok=True)


def scan_directory(split_dir: Path) -> list:
    """
    Scan a split directory and find all valid QP23/QP40 pairs.

    Returns:
        List of dicts with file information
    """
    records = []

    if not split_dir.exists():
        logger.warning(f"Directory not found: {split_dir}")
        return records

    # Find all QP23 stats files
    qp23_files = list(split_dir.rglob("*_qp23_stats.json"))
    logger.info(f"Found {len(qp23_files)} QP23 stats files in {split_dir.name}")

    for qp23_json in tqdm(qp23_files, desc=f"Scanning {split_dir.name}"):
        # Derive corresponding QP40 path
        base_name = qp23_json.stem.replace('_qp23_stats', '')
        parent_dir = qp23_json.parent

        qp40_json = parent_dir / f"{base_name}_qp40_stats.json"
        qp23_img = parent_dir / f"{base_name}_qp23.png"
        qp40_img = parent_dir / f"{base_name}_qp40.png"

        # Verify all files exist
        if not qp40_json.exists():
            logger.warning(f"Missing QP40 stats: {qp40_json}")
            continue
        if not qp23_img.exists():
            logger.warning(f"Missing QP23 image: {qp23_img}")
            continue
        if not qp40_img.exists():
            logger.warning(f"Missing QP40 image: {qp40_img}")
            continue

        # Verify JSON files are valid
        try:
            with open(qp23_json) as f:
                stats_23 = json.load(f)
            with open(qp40_json) as f:
                stats_40 = json.load(f)
        except Exception as e:
            logger.warning(f"Invalid JSON: {qp23_json} - {e}")
            continue

        # Extract metadata from path
        # Expected structure: split/method/video_id/frame_qp*.png
        parts = str(qp23_json.relative_to(OUTPUT_BASE)).split('/')
        if len(parts) >= 3:
            method = parts[1]  # e.g., Real_youtube, Deepfakes, FaceSwap
            video_id = parts[2]  # e.g., 448
        else:
            method = "unknown"
            video_id = parent_dir.name

        # Determine label
        label = 0 if 'Real' in method else 1

        # Extract frame index from filename
        try:
            frame_idx = int(base_name)
        except ValueError:
            frame_idx = 0

        records.append({
            'video_id': video_id,
            'frame_idx': frame_idx,
            'label': label,
            'method': method,
            'qp23_img_path': str(qp23_img),
            'qp40_img_path': str(qp40_img),
            'qp23_json_path': str(qp23_json),
            'qp40_json_path': str(qp40_json)
        })

    return records


def main():
    logger.info("=" * 60)
    logger.info("Building Index - Generate Master CSVs")
    logger.info("=" * 60)
    logger.info(f"Scanning: {OUTPUT_BASE}")

    # Process each split
    splits = ['train', 'val', 'test']
    stats = {}

    for split in splits:
        split_dir = OUTPUT_BASE / split
        records = scan_directory(split_dir)

        if len(records) > 0:
            df = pd.DataFrame(records)
            csv_file = CSV_PATH / f'{split}_metadata.csv'
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved {csv_file} with {len(df)} entries")

            # Compute stats
            real_count = len(df[df['label'] == 0])
            fake_count = len(df[df['label'] == 1])
            stats[split] = {
                'total': len(df),
                'real': real_count,
                'fake': fake_count,
                'methods': df['method'].value_counts().to_dict()
            }
        else:
            logger.warning(f"No valid pairs found for {split}")
            stats[split] = {'total': 0, 'real': 0, 'fake': 0}

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Index Building Complete!")
    logger.info("=" * 60)

    for split, s in stats.items():
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Total: {s['total']}")
        logger.info(f"  Real: {s['real']}")
        logger.info(f"  Fake: {s['fake']}")
        if 'methods' in s:
            logger.info(f"  Methods: {s['methods']}")

    logger.info(f"\nCSV files saved to: {CSV_PATH}")

    # Verify Cross-Generator setup
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Generator Verification")
    logger.info("=" * 60)

    train_csv = CSV_PATH / 'train_metadata.csv'
    test_csv = CSV_PATH / 'test_metadata.csv'

    if train_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        train_methods = set(train_df[train_df['label'] == 1]['method'].unique())
        test_methods = set(test_df[test_df['label'] == 1]['method'].unique())

        logger.info(f"Train fake methods: {train_methods}")
        logger.info(f"Test fake methods: {test_methods}")

        if train_methods != test_methods:
            logger.info("✓ Cross-Generator setup verified (different fake methods)")
        else:
            logger.warning("⚠ Same fake methods in train and test - not cross-generator!")


if __name__ == '__main__':
    main()
