# BitDFD 同學實作（H.264 編碼特徵 Deepfake 偵測）

重點：針對 C23 幀資料，使用 PyH264 重新編碼產生 H.264 編碼統計，並做跨生成器偵測。包含對比學習（Track A）與 H.264 統計基線（Track B）。

## 目錄與腳本
- `preprocess.py`：PyH264 重新編碼並萃取 H.264 統計與特徵。
- `build_index.py`：依處理後資料建立 train/val/test CSV 索引。
- `train.py`：Track A 對比學習訓練（ResNet18 + CQIL）。
- `eval_contrastive.py`：Track A 線性探針評估。
- `eval_nonlinear.py`：Track A 非線性分類器評估（SVM/MLP/RF）。
- `train_baseline.py`：Track B H.264 統計基線（Random Forest）。
- `ablation_double_compression.py`：雙重壓縮假設檢驗。

## 資料假設
- 幀根目錄：`/ssd6/fan/c23/<split>/<method>/<video_id>/*.png`（或有 `list.txt` 指定順序）。
- 交叉生成器協定：Train 用 Deepfakes，Val/Test 用 FaceSwap。

## 快速開始
```bash
# 1) 預處理：PyH264 重新編碼並產生特徵
python preprocess.py

# 2) 建立索引
python build_index.py

# 3) Track A 對比學習
python train.py
CUDA_VISIBLE_DEVICES=0 python eval_contrastive.py

# 4) Track B 編碼統計基線
python train_baseline.py

# 5) 選擇性實驗
python eval_nonlinear.py
python ablation_double_compression.py
```

## 產出結構
```
checkpoints/
├── pyh264_contrastive/
│   ├── best_model.pt
│   └── linear_probe_results.json
└── pyh264_baseline/
    ├── random_forest_model.joblib
    └── baseline_results.json
```

## 依賴
- Python 3.9+
- PyTorch、scikit-learn、pandas、numpy、Pillow、tqdm
- PyH264（本地模組，用於 H.264 重新編碼）
