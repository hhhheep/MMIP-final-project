#!/usr/bin/env python3
"""
PyH264 Preprocessing Script with Feature Extraction
Encodes images with H.264, extracts syntax statistics, and generates master CSVs.
Supports resume capability and robust error handling.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
import traceback
import argparse

# Add PyH264 to path
sys.path.insert(0, '/ssd5/ia313553058/MMIP/PyH264')
from h264.H264 import H264

# Paths
BASE_PATH = Path('/ssd5/ia313553058/MMIP')
C23_PATH = BASE_PATH / 'c23'
BKCHEN_CSV_PATH = C23_PATH / 'bkChen/MMIP/data/rppg_images_processed/csv'
OUTPUT_BASE = C23_PATH / 'OutPut/Offline_pre/pyh264_images_processed'
OUTPUT_CSV_PATH = OUTPUT_BASE / 'csv'
LOG_DIR = BASE_PATH / 'preprocessing_logs'
ERROR_LOG_PATH = LOG_DIR / 'errors.log'

# Ensure directories exist
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'preprocess.log')
    ]
)
logger = logging.getLogger(__name__)


def log_error(message: str):
    """Log error to dedicated error log file."""
    with open(ERROR_LOG_PATH, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def convert_path(old_path: str) -> str:
    """Convert old paths to current paths."""
    return old_path.replace('/ssd2/deepfake/c23_crop_face/', str(C23_PATH) + '/')


def extract_encoding_stats(encoder, bitstream: str, encoding_time: float, qp: int) -> dict:
    """
    Extract encoding statistics from a compressed frame.

    Args:
        encoder: H264 encoder object after compression
        bitstream: The compressed bitstream string
        encoding_time: Time taken to encode (seconds)
        qp: Quantization parameter used

    Returns:
        dict: Encoding statistics
    """
    frame = encoder.frames[0]

    # Basic metrics
    total_bits = len(bitstream)
    width = encoder.width
    height = encoder.height
    num_pixels = width * height

    # Calculate compression metrics
    # Using 8 bits per pixel (grayscale) as PyH264 processes Y channel
    original_bits = num_pixels * 8
    compression_ratio = original_bits / total_bits if total_bits > 0 else 0
    bpp = total_bits / num_pixels if num_pixels > 0 else 0

    stats = {
        'total_bits': total_bits,
        'bpp': float(bpp),
        'compression_ratio': float(compression_ratio),
        'width': width,
        'height': height,
        'encoding_time': float(encoding_time),
        'qp': qp
    }

    # Deep inspection with try-except (may not work for all PyH264 versions)
    try:
        mode_counts = {'dc': 0, 'h': 0, 'v': 0, 'other': 0}
        non_zero_coeff_count = 0
        total_coeff_count = 0
        dc_coefficients = []

        for slice_obj in frame.slices:
            for macroblock in slice_obj.blocks:
                for transform_block in macroblock.blocks:
                    # Count prediction modes
                    mode = getattr(transform_block, 'prediction_mode', None)
                    if mode in mode_counts:
                        mode_counts[mode] += 1
                    elif mode is not None:
                        mode_counts['other'] += 1

                    # Get coefficients if available
                    coeffs = getattr(transform_block, 'block', None)
                    if coeffs is not None and hasattr(coeffs, 'flatten'):
                        flat_coeffs = coeffs.flatten()
                        non_zero_coeff_count += np.count_nonzero(flat_coeffs)
                        total_coeff_count += len(flat_coeffs)
                        # DC coefficient is at position [0,0]
                        if coeffs.shape[0] > 0 and coeffs.shape[1] > 0:
                            dc_coefficients.append(float(coeffs[0, 0]))

        stats['prediction_modes'] = mode_counts
        stats['non_zero_coeff_count'] = non_zero_coeff_count
        stats['total_coeff_count'] = total_coeff_count
        stats['non_zero_ratio'] = float(non_zero_coeff_count / total_coeff_count) if total_coeff_count > 0 else 0.0

        if dc_coefficients:
            stats['dc_mean'] = float(np.mean(dc_coefficients))
            stats['dc_std'] = float(np.std(dc_coefficients))
        else:
            stats['dc_mean'] = 0.0
            stats['dc_std'] = 0.0

    except Exception as e:
        # Fallback: just log that deep inspection failed
        logger.debug(f"Deep inspection failed (this is OK): {e}")
        stats['prediction_modes'] = None
        stats['non_zero_ratio'] = None
        stats['dc_mean'] = None
        stats['dc_std'] = None

    return stats


def encode_decode_image(image_path: Path, qp: int, output_img_path: Path, output_stats_path: Path) -> tuple:
    """
    Encode and decode an image using PyH264, saving stats.

    Args:
        image_path: Path to input image
        qp: Quantization parameter (23 or 40)
        output_img_path: Path to save decoded image
        output_stats_path: Path to save encoding statistics JSON

    Returns:
        tuple: (success: bool, stats: dict or None)
    """
    try:
        # Create encoder with specified QP
        encoder = H264(qp=qp)

        # Load image
        encoder.load_image(str(image_path))

        # Measure encoding time
        start_time = time.time()
        bitstream = encoder.compress_frame(0)
        encoding_time = time.time() - start_time

        # Extract statistics BEFORE decoding
        stats = extract_encoding_stats(encoder, bitstream, encoding_time, qp)
        stats['source_path'] = str(image_path)

        # Create decoder and decode
        decoder = H264(width=encoder.width, height=encoder.height, qp=qp)
        decoder.load_bitstream(bitstream)

        # Get decoded frame
        decoded_frame = decoder.frames[0].get_image()

        # Save decoded image
        img = Image.fromarray(decoded_frame, 'L')
        img.save(str(output_img_path))

        # Safety check: verify file size > 0
        if not output_img_path.exists() or output_img_path.stat().st_size == 0:
            logger.error(f"Output image is empty or missing: {output_img_path}")
            if output_img_path.exists():
                output_img_path.unlink()
            return False, None

        # Save statistics
        with open(output_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # Safety check for stats file
        if not output_stats_path.exists() or output_stats_path.stat().st_size == 0:
            logger.error(f"Stats file is empty or missing: {output_stats_path}")
            if output_stats_path.exists():
                output_stats_path.unlink()
            return False, None

        return True, stats

    except Exception as e:
        error_msg = f"Error processing {image_path} with QP={qp}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        log_error(error_msg)
        return False, None


def filter_cross_generator(df: pd.DataFrame, split: str, method_filter: str = None) -> pd.DataFrame:
    """Filter dataframe for cross_generator mode.

    Args:
        df: Input dataframe
        split: Split name (train, val, test)
        method_filter: If specified, only include this method (e.g., 'Deepfakes', 'Real_youtube')
    """
    if split == 'train':
        # Training: Deepfakes + Real_youtube
        mask = df['rgb_path'].str.contains('Deepfakes|Real_youtube')
    else:
        # Val/Test: FaceSwap + Real_youtube
        mask = df['rgb_path'].str.contains('FaceSwap|Real_youtube')

    filtered = df[mask].copy()

    # Apply method filter if specified
    if method_filter:
        filtered = filtered[filtered['rgb_path'].str.contains(method_filter)]

    return filtered


def check_already_processed(qp23_stats_path: Path, qp40_stats_path: Path) -> bool:
    """Check if both stats files exist and are valid (>0 bytes)."""
    if not qp23_stats_path.exists() or not qp40_stats_path.exists():
        return False
    if qp23_stats_path.stat().st_size == 0 or qp40_stats_path.stat().st_size == 0:
        return False
    return True


def process_split(split_name: str, limit: int = None, method_filter: str = None) -> pd.DataFrame:
    """
    Process a single split with resume support.

    Args:
        split_name: Name of split (train, val, test)
        limit: Limit number of images to process
        method_filter: If specified, only process this method (e.g., 'Deepfakes')

    Returns:
        DataFrame with processed entries for master CSV
    """
    logger.info(f"Processing {split_name} split...")

    # Read source CSV
    csv_path = BKCHEN_CSV_PATH / f'{split_name}_metadata_multi_modal.csv'
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} entries from CSV")

    # Filter for cross_generator mode
    df_filtered = filter_cross_generator(df, split_name, method_filter)
    filter_msg = f"Filtered to {len(df_filtered)} entries for cross_generator mode"
    if method_filter:
        filter_msg += f" (method: {method_filter})"
    logger.info(filter_msg)

    # Apply limit if specified
    if limit:
        df_filtered = df_filtered.head(limit)
        logger.info(f"Limited to {len(df_filtered)} entries")

    # Convert paths
    df_filtered['rgb_path'] = df_filtered['rgb_path'].apply(convert_path)

    # Prepare output records
    output_records = []
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Process each image
    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc=f"Processing {split_name}"):
        video_id = str(row['video_id'])
        frame_idx = row['frame_idx']
        rgb_path = Path(row['rgb_path'])
        label = int(row['label'])

        # Determine output paths
        parts = str(rgb_path).split('/')
        method = parts[-3]  # e.g., Real_youtube, Deepfakes
        video_folder = parts[-2]  # e.g., 448
        base_name = rgb_path.stem  # e.g., 045

        # Create output directory
        output_dir = OUTPUT_BASE / split_name / method / video_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        qp23_img_path = output_dir / f"{base_name}_qp23.png"
        qp40_img_path = output_dir / f"{base_name}_qp40.png"
        qp23_stats_path = output_dir / f"{base_name}_qp23_stats.json"
        qp40_stats_path = output_dir / f"{base_name}_qp40_stats.json"

        # Check if already processed (resume capability)
        if check_already_processed(qp23_stats_path, qp40_stats_path):
            skipped_count += 1
            # Still add to output records
            output_records.append({
                'video_id': video_id,
                'frame_idx': frame_idx,
                'label': label,
                'method': method,
                'qp23_img_path': str(qp23_img_path),
                'qp40_img_path': str(qp40_img_path),
                'qp23_json_path': str(qp23_stats_path),
                'qp40_json_path': str(qp40_stats_path)
            })
            continue

        # Wrap each image processing in try-except
        try:
            # Check if source image exists
            if not rgb_path.exists():
                logger.warning(f"Source image not found: {rgb_path}")
                log_error(f"Source not found: {rgb_path}")
                failed_count += 1
                continue

            # Process with QP23
            success_23, stats_23 = encode_decode_image(rgb_path, qp=23,
                                                        output_img_path=qp23_img_path,
                                                        output_stats_path=qp23_stats_path)

            # Process with QP40
            success_40, stats_40 = encode_decode_image(rgb_path, qp=40,
                                                        output_img_path=qp40_img_path,
                                                        output_stats_path=qp40_stats_path)

            if success_23 and success_40:
                processed_count += 1
                output_records.append({
                    'video_id': video_id,
                    'frame_idx': frame_idx,
                    'label': label,
                    'method': method,
                    'qp23_img_path': str(qp23_img_path),
                    'qp40_img_path': str(qp40_img_path),
                    'qp23_json_path': str(qp23_stats_path),
                    'qp40_json_path': str(qp40_stats_path)
                })
            else:
                failed_count += 1

        except Exception as e:
            error_msg = f"Unexpected error for {video_id}/{base_name}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            log_error(error_msg)
            failed_count += 1
            continue

    logger.info(f"Completed {split_name}: {processed_count} processed, {skipped_count} skipped, {failed_count} failed")

    return pd.DataFrame(output_records)


def generate_master_csvs(split_dfs: dict):
    """Generate master CSV files for each split."""
    for split_name, df in split_dfs.items():
        if df is not None and len(df) > 0:
            csv_path = OUTPUT_CSV_PATH / f'{split_name}_metadata.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved master CSV: {csv_path} ({len(df)} entries)")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='PyH264 preprocessing with feature extraction')
    parser.add_argument('--limit', type=int, help='Limit number of images per split (for testing)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], help='Process only specific split')
    parser.add_argument('--method', type=str, help='Only process specific method (e.g., Deepfakes, Real_youtube, FaceSwap)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PyH264 Preprocessing with Feature Extraction")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_BASE}")
    if args.method:
        logger.info(f"Method filter: {args.method}")

    # Determine splits to process
    if args.split:
        splits = [args.split]
    else:
        splits = ['train', 'val', 'test']

    # Process each split and collect DataFrames
    split_dfs = {}
    for split in splits:
        split_dfs[split] = process_split(split, limit=args.limit, method_filter=args.method)

    # Generate master CSVs
    generate_master_csvs(split_dfs)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)
    for split_name, df in split_dfs.items():
        if df is not None:
            logger.info(f"  {split_name}: {len(df)} entries")
            if len(df) > 0:
                real_count = len(df[df['label'] == 0])
                fake_count = len(df[df['label'] == 1])
                logger.info(f"    Real: {real_count}, Fake: {fake_count}")

    logger.info(f"\nMaster CSVs saved to: {OUTPUT_CSV_PATH}")
    logger.info(f"Error log: {ERROR_LOG_PATH}")


if __name__ == '__main__':
    main()
