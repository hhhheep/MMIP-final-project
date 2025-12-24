import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import random
import argparse # 引入 argparse 處理參數
import json     # 引入 json 儲存結果
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score # 引入 AUC 計算

from model import MultiModalDetector
from dataloader import init_dataloaders 

# ============================================================
# 訓練設定
# ============================================================
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
# L2 正則化強度 (Weight Decay)
WEIGHT_DECAY = 1e-4 # <--- 引入 L2 正則化策略
# 早期停止參數
PATIENCE = 5 # <--- 連續 5 個 epoch 性能未改善則停止
MODAL_DROPOUT_RATE = 0.2 # <--- 模態 Dropout 概率 (隨機禁用一個模態)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cuda:0")
# DEVICE = torch.device("cpu")
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# 訓練函數 (保持不變，因為訓練時不需要計算 AUC)
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, epoch, writer, modality_mode):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", unit="batch")
    
    for I_rgb, I_rppg, labels in pbar:
        I_rgb = I_rgb.to(DEVICE)
        I_rppg = I_rppg.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        # --- 應用模態 Dropout (僅在 multi_modal 模式下) ---
        input_rgb = I_rgb
        input_rppg = I_rppg
        
        if modality_mode == 'multi_modal' and random.random() < MODAL_DROPOUT_RATE:
            # 隨機禁用一個模態
            if random.random() < 0.5:
                # 禁用 RGB
                input_rgb = I_rgb.new_zeros(I_rgb.shape)
            else:
                # 禁用 rPPG
                input_rppg = I_rppg.new_zeros(I_rppg.shape)


        # 根據模態模式選擇輸入
        if modality_mode == 'rgb_only':
            outputs = model(I_rgb, I_rgb.new_zeros(I_rppg.shape)) # 傳入零張量禁用 rPPG 路徑
        elif modality_mode == 'rppg_only':
            outputs = model(I_rgb.new_zeros(I_rgb.shape), I_rppg) # 傳入零張量禁用 RGB 路徑
        else: # multi_modal
            outputs = model(I_rgb, I_rppg)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/Train', avg_loss, epoch)
    print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}")
    
    return avg_loss

# ============================================================
# 驗證函數 (新增 AUC 計算與模態切換)
# ============================================================
def validate_epoch(model, dataloader, criterion, epoch, writer, modality_mode, set_name):
    model.eval()
    total_loss = 0.0
    
    all_labels = []
    all_scores = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{set_name}]", unit="batch")
    
    with torch.no_grad():
        for I_rgb, I_rppg, labels in pbar:
            I_rgb = I_rgb.to(DEVICE)
            I_rppg = I_rppg.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            # 根據模態模式選擇輸入
            if modality_mode == 'rgb_only':
                outputs = model(I_rgb, I_rgb.new_zeros(I_rppg.shape))
            elif modality_mode == 'rppg_only':
                outputs = model(I_rgb.new_zeros(I_rgb.shape), I_rppg)
            else: # multi_modal
                outputs = model(I_rgb, I_rppg)
                
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 將 logits 轉換為概率分數
            scores = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores)
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # 計算最終指標
    avg_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 預測類別 (閾值 0.5)
    all_preds = (all_scores > 0.5).astype(int) 
    
    # 計算 AUC
    auc_score = roc_auc_score(all_labels, all_scores)
    
    # 計算 Accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # TensorBoard 可視化
    writer.add_scalar(f'Loss/{set_name}', avg_loss, epoch)
    writer.add_scalar(f'Accuracy/{set_name}', accuracy, epoch)
    writer.add_scalar(f'AUC/{set_name}', auc_score, epoch)
    
    print(f"Epoch {epoch} {set_name} Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, AUC: {auc_score:.4f}")
    
    return avg_loss, accuracy, auc_score

# ============================================================
# 主程式 (新增參數解析)
# ============================================================
def main(modality_mode, train_mode): # main 函數現在接收兩個模式參數
    # 設定 TensorBoard 日誌目錄名稱，包含兩種模式和時間戳記
    log_time = time.strftime("%Y%m%d-%H%M%S")
    global LOG_DIR, WRITER
    
    # 設置 LOG_DIR：例如 runs/multi_modal_cross_generator_20251129-215000
    LOG_DIR = f'runs/stage3/{modality_mode}_{train_mode}_{log_time}'
    WRITER = SummaryWriter(LOG_DIR)
    
    # --- 1. 初始化 DataLoader ---
    global train_loader, val_loader
    train_loader, val_loader = init_dataloaders(train_mode)
    
    # 2. 初始化模型 (保持不變)
    model = MultiModalDetector(num_classes=1)    # 根據模態模式凍結不使用的分支 (可選，但更嚴謹)
    if modality_mode == 'rgb_only':
        print("模式: 僅使用 RGB。凍結 rPPG 相關參數。")
        for param in model.E_rppg.parameters():
            param.requires_grad = False
    elif modality_mode == 'rppg_only':
        print("模式: 僅使用 rPPG。凍結 RGB 相關參數。")
        for param in model.E_rgb.parameters():
            param.requires_grad = False
    
    model.to(DEVICE)
    
    # 2. 定義損失函數和優化器
    criterion = nn.BCEWithLogitsLoss() 
    
    # 優化器只更新 requires_grad=True 的參數
    # 賦予 Adam 優化器 Weight Decay 參數
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY) # <--- L2 正則化應用 
       
    best_val_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    # 早期停止計數器
    epochs_no_improve = 0

    print(f"Starting training for mode: {modality_mode} on device: {DEVICE}")
    
    # 3. 訓練循環
    for epoch in range(1, NUM_EPOCHS + 1):
        # 訓練階段
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, WRITER, modality_mode)
        
        # 驗證階段
        val_loss, val_acc, val_auc = validate_epoch(model, val_loader, criterion, epoch, WRITER, modality_mode, 'Validation')
        
        # 5. 早期停止檢查
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0 # 重置計數器
            
            # 儲存模型 (路徑中加入 mode 和 log_time)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{modality_mode}_{train_mode}_best_model_{log_time}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path} (Best Val AUC: {best_val_auc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"No improvement in Val AUC. Counter: {epochs_no_improve}/{PATIENCE}")
            
        if epochs_no_improve == PATIENCE:
            print(f"--- 🛑 Early stopping triggered after {PATIENCE} epochs without improvement. ---")
            break # 跳出訓練循環


        # 4. 儲存訓練歷史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # 5. 儲存模型 (根據 AUC)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{modality_mode}_best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path} (Best Val AUC: {best_val_auc:.4f})")

        print("-" * 50)

    WRITER.close()
    print("Training finished.")
    
    # 6. 將訓練歷史儲存到檔案
    history_file = os.path.join(CHECKPOINT_DIR, f'{modality_mode}_training_history.json')
    with open(history_file, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {history_file}")


# ============================================================
# 執行器 (使用 argparse 處理參數切換)
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Modal Deepfake Detector Training")
    
    # 模態模式 (已存在)
    parser.add_argument('--mode', type=str, default='rppg_only', 
                        choices=['rgb_only', 'rppg_only', 'multi_modal'],
                        help="選擇運行的模態: rgb_only, rppg_only, 或 multi_modal")
    
    # 資料集過濾模式 (新增)
    parser.add_argument('--train_mode', type=str, default='unrestricted',
                        choices=['unrestricted', 'cross_generator'],
                        help="資料集過濾模式: unrestricted (所有數據), cross_generator (Train: Deepfakes, Val/Test: FaceSwap)")
    
    args = parser.parse_args()
    
    # 傳遞兩個參數給 main 函數
    main(args.mode, args.train_mode)

# python /ssd1/bkchen/MMIP/scripts/model/train.py --mode rppg_only --train_mode cross_generator