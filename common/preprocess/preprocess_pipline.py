"""
簡化版前處理 pipeline（NoiseDF-lite）：
- 抽灰階
- 臉框偵測
- 臉 / 背景 patch 裁切
- anti-aliasing resize 到 128×128
- 單尺度 Gaussian 殘差（給可視化 / hand-crafted）
- 多尺度殘差 + LoG 堆疊（給孿生 ResNet-18）
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from scipy import ndimage

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    _HAS_MEDIAPIPE = False

try:
    import dlib
    _HAS_DLIB = True
except ImportError:
    dlib = None
    _HAS_DLIB = False

try:
    import cv2
except ImportError as e:
    raise RuntimeError("cv2 (OpenCV) is required for I/O") from e


@dataclass
class PatchSample:
    frame_id: int
    face_patch: np.ndarray      # 單尺度殘差，給可視化 / 傳統特徵
    bg_patch: np.ndarray
    label: int                  # 1 fake, 0 real
    face_raw: Optional[np.ndarray] = None  # (H,W) 0~1，給 multi-scale
    bg_raw: Optional[np.ndarray] = None


# ---------- 基礎影像工具 ----------

def to_gray(frame: np.ndarray) -> np.ndarray:
    """RGB → gray，輸出 float32（0~1）。"""
    if frame.ndim == 2:
        gray = frame.astype(np.float32)
    else:
        r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray.astype(np.float32), 0.0, 255.0)
    # keep scaling in one place to avoid guessing downstream
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def crop_square(gray: np.ndarray, center: Tuple[int, int], side: int) -> np.ndarray:
    """以中心與邊長裁剪正方形 patch。"""
    h, w = gray.shape
    cx, cy = center
    half = side // 2
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    return gray[y0:y1, x0:x1]


def resize_to(patch: np.ndarray, size: int = 128) -> np.ndarray:
    """使用 OpenCV resize，並維持 0~1 float32 值域：
       - 下採樣：INTER_AREA（內建 anti-alias）
       - 上採樣：INTER_CUBIC
    """
    h, w = patch.shape
    patch_f = np.clip(patch.astype(np.float32), 0.0, 255.0)
    if patch_f.max() > 1.0:
        patch_f = patch_f / 255.0
    if h > size or w > size:
        out = cv2.resize(patch_f, (size, size), interpolation=cv2.INTER_AREA)
    else:
        out = cv2.resize(patch_f, (size, size), interpolation=cv2.INTER_CUBIC)
    return out.astype(np.float32)


# ---------- 臉框偵測 ----------

def _detect_face_mediapipe(gray: np.ndarray) -> Tuple[Tuple[int, int], int]:
    if not _HAS_MEDIAPIPE:
        raise ValueError("mediapipe not available")
    h, w = gray.shape
    gray_u8 = gray if gray.dtype == np.uint8 else np.clip(gray * (255.0 if gray.max() <= 1.5 else 1.0), 0.0, 255.0)
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.uint8)
    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as fd:
        res = fd.process(rgb)
    if not res.detections:
        raise ValueError("no face detected")
    best = max(
        res.detections,
        key=lambda d: d.location_data.relative_bounding_box.width
        * d.location_data.relative_bounding_box.height,
    )
    rbb = best.location_data.relative_bounding_box
    cx = int((rbb.xmin + rbb.width / 2.0) * w)
    cy = int((rbb.ymin + rbb.height / 2.0) * h)
    side = int(max(rbb.width * w, rbb.height * h))
    side = max(side, 32)
    return (cx, cy), side


def _detect_face_dlib(gray: np.ndarray) -> Tuple[Tuple[int, int], int]:
    if not _HAS_DLIB:
        raise ValueError("dlib not available")
    detector = dlib.get_frontal_face_detector()
    gray_u8 = gray if gray.dtype == np.uint8 else np.clip(gray * (255.0 if gray.max() <= 1.5 else 1.0), 0.0, 255.0)
    img = gray_u8.astype(np.uint8)
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError("no face detected")
    rect = max(dets, key=lambda r: r.width() * r.height())
    cx = (rect.left() + rect.right()) // 2
    cy = (rect.top() + rect.bottom()) // 2
    side = max(rect.width(), rect.height())
    side = max(side, 32)
    return (cx, cy), side


def _detect_face_brightness(gray: np.ndarray, guess_size: int = 64, stride: int = 16) -> Tuple[Tuple[int, int], int]:
    """沒有 mediapipe/dlib 時的 fallback：找亮度最高區塊。"""
    h, w = gray.shape
    best_score, best_center = -1.0, (w // 2, h // 2)
    for y in range(0, max(1, h - guess_size + 1), stride):
        for x in range(0, max(1, w - guess_size + 1), stride):
            patch = gray[y : y + guess_size, x : x + guess_size]
            score = float(patch.mean())
            if score > best_score:
                best_score = score
                best_center = (x + guess_size // 2, y + guess_size // 2)
    return best_center, guess_size


def detect_face_bbox(gray: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """優先 mediapipe，其次 dlib，最後 brightness heuristic。"""
    for fn in (_detect_face_mediapipe, _detect_face_dlib, _detect_face_brightness):
        try:
            return fn(gray)
        except Exception:
            continue
    return _detect_face_brightness(gray)


# ---------- 背景 patch 選取（重點：與臉同 domain） ----------

def _select_background_patch_smart(
    gray: np.ndarray,
    face_center: Tuple[int, int],
    side: int,
) -> np.ndarray:
    cx_face, cy_face = face_center
    h, w = gray.shape

    candidates = [
        (cx_face, cy_face + int(1.2 * side)),  # 下方（脖子/肩膀）優先
        (cx_face, cy_face - int(1.2 * side)),  # 上方
        (cx_face - int(1.2 * side), cy_face),  # 左
        (cx_face + int(1.2 * side), cy_face),  # 右
    ]

    for cx, cy in candidates:
        if side // 2 + 10 < cx < w - side // 2 - 10 and \
           side // 2 + 10 < cy < h - side // 2 - 10:
            patch = crop_square(gray, (cx, cy), side)
            mean_val = patch.mean()
            std_val = patch.std()
            # 避免純黑 / 純白 / 極度高頻區；值域已是 0~1
            if 0.08 < mean_val < 0.94 and 0.02 < std_val < 0.24:
                return patch

    # 若都不行，用對角線遠端當背景
    alt_center = (
        min(w - side // 2, max(side // 2, w - cx_face)),
        min(h - side // 2, max(side // 2, h - cy_face)),
    )
    return crop_square(gray, alt_center, side)


def select_background_patch(gray: np.ndarray, face_center: Tuple[int, int], side: int) -> np.ndarray:
    """背景 patch：優先 smart，失敗再隨機 fallback。"""
    h, w = gray.shape
    cx_face, cy_face = face_center

    try:
        return _select_background_patch_smart(gray, face_center, side)
    except Exception:
        pass

    def iou(box1, box2) -> float:
        x0 = max(box1[0], box2[0])
        y0 = max(box1[1], box2[1])
        x1 = min(box1[2], box2[2])
        y1 = min(box1[3], box2[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    face_box = (
        cx_face - side // 2,
        cy_face - side // 2,
        cx_face + side // 2,
        cy_face + side // 2,
    )

    # 隨機嘗試幾次不重疊的 patch
    for _ in range(50):
        cx = np.random.randint(side // 2, max(side // 2 + 1, w - side // 2))
        cy = np.random.randint(side // 2, max(side // 2 + 1, h - side // 2))
        box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
        if iou(face_box, box) < 0.05:
            return crop_square(gray, (cx, cy), side)

    # 最後 fallback
    alt_center = (
        min(w - side // 2, max(side // 2, w - cx_face)),
        min(h - side // 2, max(side // 2, h - cy_face)),
    )
    return crop_square(gray, alt_center, side)


# ---------- 殘差 & multi-scale stack ----------

def compute_residual(gray_01: np.ndarray) -> np.ndarray:
    """
    基礎 Gaussian 殘差（單一 scale）。
    gray_01: 0~1。
    """
    side = min(gray_01.shape)
    sigma_dyn = 0.02 * side
    sigma = float(np.clip(sigma_dyn, 1.0, 3.0))
    blur = ndimage.gaussian_filter(gray_01, sigma=sigma)
    return gray_01 - blur


def build_multiscale_residual_stack(gray_01: np.ndarray) -> np.ndarray:
    """
    NoiseDF-lite 多尺度殘差：
    輸入：灰階 0~1，shape=(H,W)
    輸出：shape=(3,H,W)，分別為：
      0: 小 sigma 殘差（細節）
      1: 大 sigma 殘差（粗結構）
      2: LoG edge
    """
    g = gray_01.astype(np.float32)
    h, w = g.shape
    side = min(h, w)

    sigma1 = float(np.clip(0.01 * side, 1.0, 2.0))
    sigma2 = float(np.clip(0.06 * side, 3.0, 6.0))

    base1 = ndimage.gaussian_filter(g, sigma=sigma1)
    base2 = ndimage.gaussian_filter(g, sigma=sigma2)

    res1 = g - base1
    res2 = g - base2

    base_for_lap = ndimage.gaussian_filter(g, sigma=1.0)
    lap = ndimage.laplace(base_for_lap)

    stack = np.stack([res1, res2, lap], axis=0)

    for c in range(stack.shape[0]):
        ch = stack[c]
        # 先做簡單的百分位裁切，避免極端值主導 std
        lo, hi = np.percentile(ch, [1.0, 99.0])
        ch = np.clip(ch, lo, hi)
        m = float(ch.mean())
        s = float(ch.std() + 1e-6)
        ch = (ch - m) / s
        ch = np.clip(ch, -5.0, 5.0)
        stack[c] = ch

    return stack.astype(np.float32)


# ---------- 單張 frame 的主流程 ----------

def preprocess_frame(frame: np.ndarray, frame_id: int, label: int) -> PatchSample:
    """
    主前處理：
      frame → gray → 臉 / 背景 patch → 128×128 → 0~1 → 殘差
    """
    gray = to_gray(frame)
    face_center, face_size = detect_face_bbox(gray)

    scale = 1.3
    max_side = int(0.8 * min(gray.shape))
    side = min(int(scale * face_size), max_side)

    face_patch = crop_square(gray, face_center, side)
    bg_patch = select_background_patch(gray, face_center, side)

    face_128 = resize_to(face_patch, 128)
    bg_128 = resize_to(bg_patch, 128)

    face_res = compute_residual(face_128)
    bg_res = compute_residual(bg_128)

    return PatchSample(
        frame_id=frame_id,
        face_patch=face_res,
        bg_patch=bg_res,
        label=label,
        face_raw=face_128,
        bg_raw=bg_128,
    )


def build_siamese_tensors(sample: PatchSample) -> Tuple[np.ndarray, np.ndarray]:
    """
    產生給孿生 ResNet-18 用的輸入：
      face_tensor, bg_tensor: shape=(3,128,128)
    """
    if sample.face_raw is None or sample.bg_raw is None:
        raise ValueError("PatchSample 缺少 face_raw/bg_raw")
    face_tensor = build_multiscale_residual_stack(sample.face_raw)
    bg_tensor = build_multiscale_residual_stack(sample.bg_raw)
    return face_tensor, bg_tensor


# ---------- I/O & 批次處理 ----------

def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"讀取失敗: {path}")
    return img.astype(np.float32)


def save_residual_png(residual: np.ndarray, path: str) -> None:
    res = residual - residual.min()
    denom = res.max() if res.max() > 0 else 1.0
    res = res / denom * 255.0
    res = np.clip(res, 0, 255).astype(np.uint8)
    cv2.imwrite(path, res)


def process_paths(
    paths: List[Path],
    output_dir: str,
    label: int = 0,
    prefix: str = "",
) -> List[str]:
    """
    給定一組影像路徑：
      - 輸出 face/bg 殘差 PNG
      - 輸出 face/bg multi-scale tensor (.npy)
    回傳每個樣本的 tag（方便對應）。
    """
    os.makedirs(output_dir, exist_ok=True)
    tags: List[str] = []

    def build_name(p: Path) -> str:
        parent = p.parent.name
        grand = p.parent.parent.name if p.parent.parent else ""
        tag = "__".join([x for x in [grand, parent, p.stem] if x])
        return f"{prefix}{tag}"

    for i, p in enumerate(paths):
        frame = load_image_gray(str(p))
        sample = preprocess_frame(frame, frame_id=i, label=label)
        face_tensor, bg_tensor = build_siamese_tensors(sample)

        base = build_name(p)
        tags.append(base)

        # 殘差可視化
        save_residual_png(sample.face_patch, os.path.join(output_dir, f"{base}_face_res.png"))
        save_residual_png(sample.bg_patch, os.path.join(output_dir, f"{base}_bg_res.png"))

        # CNN 輸入
        np.save(os.path.join(output_dir, f"{base}_face_tensor.npy"), face_tensor)
        np.save(os.path.join(output_dir, f"{base}_bg_tensor.npy"), bg_tensor)

    return tags
