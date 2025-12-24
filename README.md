# 多模態 Deepfake 偵測訓練流程說明

本專案包含 Noise / rPPG / RGB 三個分支，以及最終的三合一多模態模型。以下是一條相對穩定的執行路徑與常見坑位提醒。

## 前置需求
- 影像幀與 bbox：`/ssd6/fan/c23/<split>/<method>/<video_id>`，每幀旁已有 `*.bbox.json`。若換資料根目錄，`manifest.jsonl` 的 `frames_root` 需同步修改。
- rPPG cache：`runs/rppg_cache/<split>/<method>/<video_id>/<hash>.png + .meta.json`，由 `rppg/dataset_with_meta.py` 產生。
- 預訓練權重：
  - Noise: 使用者需自行提供 `noise_metric_resnet18*.pt` 路徑。
  - rPPG: 使用者需自行提供 `rppg_resnet18*.pt` 路徑。
  - RGB: 使用者需自行提供 `best_rgb_resnet18.pt` 路徑。
- bbox cache（可選）：`bbox_cache/`，可由 sidecar 框直接填充或偵測補齊。
--原始的Resnet-18 ： https://huggingface.co/microsoft/resnet-18
## 推薦執行順序
1) **（可跳過）預熱 bbox 與取樣計畫**  
   - 直接用幀旁 sidecar 框：不需額外步驟。  
   - 若要填充/固化 bbox_cache + 計畫，先跑：  
     ```bash
     python scripts/precompute_bboxes.py \
       --manifest manifest.jsonl \
       --bbox-cache bbox_cache \
       --rppg-cache runs/rppg_cache \
       --frames-per-video 8 \
       --strategy uniform
     ```
     （已有 plan 可加 `--plan-root runs/mm_plans`）

2) **Noise 分支訓練**  
   - 主要腳本：`noise/train_noise_siamese_resnet_v2.py`（或 supcon 版）。  
   - 輸出：`noise/runs_noise_supcon/.../noise_metric_resnet18_*.pt`。

3) **rPPG 分支訓練**  
   - 前處理：`rppg/dataset_with_meta.py` 生成 spectrogram + meta。  
   - 訓練腳本：`rppg/train_rppg_resnet.py`，輸出 `rppg/runs_rppg/rppg_resnet18_*.pt`。

4) **RGB 分支訓練**  
   - 腳本：`rgb/train_rgb_resnet.py`，輸出 `rgb/runs_rgb/best_rgb_resnet18.pt`。

5) **三合一多模態訓練**  
   基本指令範例（調整路徑/批次/epochs 依硬體）：
   ```bash
   python scripts/train_multimodal.py \
     --manifest manifest.jsonl \
     --noise-ckpt <path/to/noise_metric_resnet18*.pt> \
     --rppg-ckpt <path/to/rppg_resnet18*.pt> \
     --rgb-ckpt <path/to/best_rgb_resnet18.pt> \
     --rppg-cache runs/rppg_cache \
     --plan-root runs/mm_plans \
     --bbox-cache bbox_cache \
     --frames-per-video 8 \
     --batch-size 4 \
     --epochs 10
   ```
   - `--freeze-rgb` 預設 True，`--finetune-rppg` 預設 True；依需求調整。  
   - manifest 不存在時會自動掃 `/ssd6/fan/c23` 生成；若資料路徑不同請手動建立 manifest。  
   - 如果自帶計畫/框缺失導致 batch 為空，需在 train loop 增加 `if batch is None: continue`（eval 已處理）。

## 常見反直覺點
- **dlib 依賴**：`FaceTrackCache` 初始化會要求 dlib；若僅想重用已存在的 `*.bbox.json`，可在無 dlib 環境下改成 detector=None。  
- **路徑寫死**：多處預設為 `/ssd6/fan/final_project/...`，換機或路徑時記得用參數覆蓋。  
- **空 batch**：`collate` 可能因缺框/空張量丟棄整個 batch。eval loop 有防護，train loop 須自行加 `if batch is None: continue`。  
- **rPPG 最短長度**：rPPG 前處理要求至少 16 幀；過短影片會被跳過，請確保幀數充足或調整邏輯。  
- **檢查幀順序**：若存在 `list.txt` 會依檔案列順序取樣，否則 glob；若順序很重要請提供 list.txt。

## 環境（py_torch_cu124）
- Python 3.12.9，CUDA 12.4 stack：`torch 2.6.0`、`torchvision 0.21.0`、`triton 3.2.0`、`nvidia-cudnn-cu12 9.1.0` 及對應 CUDA runtime/cuBLAS/cuDNN/cuFFT/cuRAND/cuSolver/cuSparse/NCCL。
- 影像與增強：`opencv-python 4.6.0.66`、`opencv-python-headless 4.11.0.86`、`albumentations 1.1.0`、`imgaug 0.4.0`、`Pillow 11.1.0`、`scikit-image 0.25.2`。
- 模型與特徵：`timm 0.6.12`、`efficientnet-pytorch 0.7.1`、`segmentation-models-pytorch 0.3.2`、`pretrainedmodels 0.7.4`。
- NLP/HF：`transformers 4.50.1`、`tokenizers 0.21.1`、`huggingface-hub 0.29.3`、`safetensors 0.5.3`。
- 數值/學習：`numpy 2.2.4`、`scikit-learn 1.6.1`、`scipy 1.15.2`、`pandas 1.4.2`、`tensorboard 2.10.1`。
- 其他：`tqdm 4.61.0`、`rich 13.9.4`。
- bbox 偵測若需 dlib，請自行安裝（目前環境未安裝，僅讀取 sidecar bbox 或 cache）。

## 快速檢查
- `manifest.jsonl` 的 `frames_root` 是否對應實際幀路徑。  
- `runs/rppg_cache` 是否存在對應 split/method/video 的 spectrogram 和 meta。  
- 三個分支 ckpt 路徑是否正確。  
- bbox 是否可讀（sidecar 或 bbox_cache）。  
- 顯存不足時可降低 `--batch-size` 或 `--frames-per-video`。

## 其他專案：BitDFD 同學實作
- 路徑：`BitDFD/BitDFD`（已解壓縮）。請閱讀其中的 `README.md` 以獲得該專案的流程與指令。  
- 主要針對 C23 幀做 PyH264 重新編碼，訓練對比學習與 H.264 統計基線。
