from torch.utils.data import DataLoader
import argparse

from dataset import VideoRPPGDataset

# 參數設定
BATCH_SIZE = 32
NUM_WORKERS = 4

# 已裁好的 frame 根目錄
FRAMES_ROOT = '/ssd2/deepfake/c23'

# ============================================================
# 函數：初始化 DataLoader
# ============================================================
def init_dataloaders(train_mode,
                     frames_root: str = FRAMES_ROOT,
                     rppg_cache_dir: str = 'rppg_cache',
                     rppg_window_size: int = 48,
                     rppg_fps: int = 30):
    """
    直接從 frame 生成代表影片的 rPPG，不依賴既有 rppg_path。
    """
    train_allowed = None
    val_allowed = None
    if train_mode == 'cross_generator':
        train_allowed = ['Deepfakes', 'Real_youtube']
        val_allowed = ['FaceSwap', 'Real_youtube']

    train_dataset = VideoRPPGDataset(
        frames_root=frames_root,
        split='train',
        frames_per_video=None,
        frame_sample="uniform",
        allowed_methods=train_allowed,
        rppg_cache_dir=rppg_cache_dir,
        rppg_window_size=rppg_window_size,
        rppg_fps=rppg_fps,
    )

    val_dataset = VideoRPPGDataset(
        frames_root=frames_root,
        split='val',
        frames_per_video=None,
        frame_sample="uniform",
        allowed_methods=val_allowed,
        rppg_cache_dir=rppg_cache_dir,
        rppg_window_size=rppg_window_size,
        rppg_fps=rppg_fps,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"\n--- DataLoader Summary (Mode: {train_mode}) ---")
    print(f"訓練集影片數: {len(train_dataset)}")
    print(f"驗證集影片數: {len(val_dataset)}")
    print(f"訓練批次數: {len(train_loader)}")
    print("-" * 30)

    return train_loader, val_loader


# ============================================================
# 執行測試邏輯 (此部分不會在 train.py 載入時運行)
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DataLoader Test")
    parser.add_argument('--train_mode', type=str, default='unrestricted',
                        choices=['unrestricted', 'cross_generator'],
                        help="資料集過濾模式")
    parser.add_argument('--frames_root', type=str, default=FRAMES_ROOT)
    parser.add_argument('--rppg_cache_dir', type=str, default='rppg_cache')
    parser.add_argument('--rppg_window_size', type=int, default=48)
    parser.add_argument('--rppg_fps', type=int, default=30)

    args = parser.parse_args()

    train_loader, val_loader = init_dataloaders(args.train_mode,
                                               frames_root=args.frames_root,
                                               rppg_cache_dir=args.rppg_cache_dir,
                                               rppg_window_size=args.rppg_window_size,
                                               rppg_fps=args.rppg_fps)

    # 測試 DataLoader
    for batch_idx, (I_rgb, I_rppg, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  I_rgb 批次尺寸: {I_rgb.shape}")
        print(f"  I_rppg 批次尺寸: {I_rppg.shape}")
        print(f"  Labels 批次尺寸: {labels.shape}")
        break
