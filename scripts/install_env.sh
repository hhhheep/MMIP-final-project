#!/usr/bin/env bash
# One-click environment setup for this project.
# Assumes you are already inside the desired Python environment (e.g., conda env with Python 3.12).
# Installs PyTorch CUDA 12.4 stack plus project dependencies.

set -euo pipefail

echo "[env] Using python: $(which python)"
python --version

echo "[env] Upgrading pip..."
python -m pip install --upgrade pip

echo "[env] Installing PyTorch CUDA 12.4 stack..."
python -m pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --extra-index-url https://download.pytorch.org/whl/cu124

echo "[env] Installing core dependencies..."
python -m pip install \
  numpy==2.2.4 \
  scipy==1.15.2 \
  scikit-learn==1.6.1 \
  pandas==1.4.2 \
  tensorboard==2.10.1 \
  tqdm==4.61.0 \
  rich==13.9.4

echo "[env] Installing vision + augmentation libs..."
python -m pip install \
  opencv-python==4.6.0.66 \
  opencv-python-headless==4.11.0.86 \
  pillow==11.1.0 \
  albumentations==1.1.0 \
  imgaug==0.4.0 \
  scikit-image==0.25.2

echo "[env] Installing model helpers..."
python -m pip install \
  timm==0.6.12 \
  efficientnet-pytorch==0.7.1 \
  segmentation-models-pytorch==0.3.2 \
  pretrainedmodels==0.7.4

echo "[env] Installing HF + serialization..."
python -m pip install \
  transformers==4.50.1 \
  tokenizers==0.21.1 \
  huggingface-hub==0.29.3 \
  safetensors==0.5.3

echo "[env] Optional: install dlib if you need on-the-fly bbox detection (sidecar bbox 不需安裝)。"
echo "       sudo apt-get install -y libboost-all-dev || true"
echo "       python -m pip install dlib"

echo "[env] Done. Verify with: python - <<'PY'\nimport torch, torchvision; print(torch.__version__, torch.cuda.is_available()); print(torchvision.__version__)\nPY"
