"""
簡易前處理 demo：
- 模擬抽帧、臉/背景裁切、殘差與 DCT、手工特徵與關係特徵。
- 僅依賴 numpy / scipy，使用合成影像示範流程，方便在無影像庫時快速跑通。
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from scipy.fftpack import dct

try:
    import mediapipe as mp  # type: ignore

    _HAS_MEDIAPIPE = True
except ImportError:  # pragma: no cover - optional dependency
    mp = None
    _HAS_MEDIAPIPE = False

try:
    import dlib  # type: ignore

    _HAS_DLIB = True
except ImportError:  # pragma: no cover - optional dependency
    dlib = None
    _HAS_DLIB = False

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except ImportError:  # pragma: no cover - optional dependency
    # 環境應已具備，若缺請自行安裝 opencv-python
    raise


@dataclass
class PatchSample:
    frame_id: int
    face_patch: np.ndarray
    bg_patch: np.ndarray
    label: int  # 1 fake, 0 real


def synthetic_frame(h: int = 256, w: int = 256, face_scale: float = 0.25) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """生成一張隨機影像並塞入一個較亮的方塊當臉，回傳影像、臉中心與尺寸。"""
    frame = np.random.rand(h, w, 3) * 40 + 90  # 稍微均勻的背景
    face_size = int(min(h, w) * face_scale)
    cy = np.random.randint(face_size, h - face_size)
    cx = np.random.randint(face_size, w - face_size)
    y0, y1 = cy - face_size // 2, cy + face_size // 2
    x0, x1 = cx - face_size // 2, cx + face_size // 2
    frame[y0:y1, x0:x1] += 80  # 臉更亮
    noise = np.random.randn(h, w, 3) * 3
    frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    return frame, (cx, cy), face_size


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32)


def _detect_face_mediapipe(gray: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """使用 mediapipe face_detection，返回中心與邊長；若失敗則丟出 ValueError。"""
    if not _HAS_MEDIAPIPE:
        raise ValueError("mediapipe not available")
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as fd:
        res = fd.process(rgb)
    if not res.detections:
        raise ValueError("no face detected")
    # 取最大框
    best = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
    rbb = best.location_data.relative_bounding_box
    cx = int((rbb.xmin + rbb.width / 2.0) * w)
    cy = int((rbb.ymin + rbb.height / 2.0) * h)
    side = int(max(rbb.width * w, rbb.height * h))
    side = max(side, 32)  # 最小邊長
    return (cx, cy), side


def _detect_face_dlib(gray: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """使用 dlib HOG frontal face detector，返回中心與邊長；若失敗則丟出 ValueError。"""
    if not _HAS_DLIB:
        raise ValueError("dlib not available")
    detector = dlib.get_frontal_face_detector()
    img = gray.astype(np.uint8)
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError("no face detected")
    # 選最大框
    rect = max(dets, key=lambda r: r.width() * r.height())
    cx = (rect.left() + rect.right()) // 2
    cy = (rect.top() + rect.bottom()) // 2
    side = max(rect.width(), rect.height())
    side = max(side, 32)
    return (cx, cy), side


def _detect_face_brightness(gray: np.ndarray, guess_size: int = 64, stride: int = 16) -> Tuple[Tuple[int, int], int]:
    """無依賴的簡易臉框偵測：找亮度最高的窗口當作臉。"""
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
    """優先 dlib，其次 mediapipe，最後退回亮度 heuristic。"""
    for fn in (_detect_face_dlib, _detect_face_mediapipe, _detect_face_brightness):
        try:
            return fn(gray)
        except Exception:
            continue
    # 理論上 _detect_face_brightness 不會丟例外，這裡僅作保險
    return _detect_face_brightness(gray)


def crop_square(gray: np.ndarray, center: Tuple[int, int], side: int) -> np.ndarray:
    """以中心與邊長裁剪正方形並自動裁掉邊界外的部分。"""
    h, w = gray.shape
    cx, cy = center
    half = side // 2
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    patch = gray[y0:y1, x0:x1]
    return patch


def select_background_patch(gray: np.ndarray, face_center: Tuple[int, int], side: int, grid: int = 4) -> np.ndarray:
    h, w = gray.shape
    cx_face, cy_face = face_center

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

    # 嘗試隨機挑一個不重疊臉的背景 patch，保持與臉同尺寸
    for _ in range(50):
        cx = np.random.randint(side // 2, max(side // 2 + 1, w - side // 2))
        cy = np.random.randint(side // 2, max(side // 2 + 1, h - side // 2))
        box = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)
        if iou(face_box, box) < 0.05:
            return crop_square(gray, (cx, cy), side)

    # 若隨機取樣都重疊太多，退回臉對角線的遠端
    alt_center = (min(w - side // 2, max(side // 2, w - cx_face)), min(h - side // 2, max(side // 2, h - cy_face)))
    return crop_square(gray, alt_center, side)


def resize_to(patch: np.ndarray, size: int = 128) -> np.ndarray:
    h, w = patch.shape
    zoom_y = size / h
    zoom_x = size / w
    # bicubic 能減少鋸齒假訊號
    return ndimage.zoom(patch, (zoom_y, zoom_x), order=3)


def compute_residual(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    # sigma 隨 patch 邊長調整，避免臉大小造成殘差對比失衡
    side = min(gray.shape)
    sigma_dyn = 0.02 * side
    sigma = float(np.clip(sigma_dyn, 1.0, 3.0))
    blur = ndimage.gaussian_filter(gray, sigma=sigma)
    return gray - blur


def block_dct_energy(patch: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float, float]:
    h, w = patch.shape
    h8, w8 = h - h % 8, w - w % 8
    patch = patch[:h8, :w8]
    blocks = patch.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
    energies = []
    for blk in blocks:
        coeff = dct(dct(blk.T, norm="ortho").T, norm="ortho")
        energies.append(np.abs(coeff) ** 2)
    energy = np.mean(energies, axis=0)

    low_mask = np.zeros((8, 8), dtype=bool)
    mid_mask = np.zeros((8, 8), dtype=bool)
    high_mask = np.zeros((8, 8), dtype=bool)
    for u in range(8):
        for v in range(8):
            if u == 0 and v == 0:
                continue  # DC
            if u + v <= 2:
                low_mask[u, v] = True
            elif 3 <= u + v <= 5:
                mid_mask[u, v] = True
            else:
                high_mask[u, v] = True

    low = float(np.sum(energy[low_mask]))
    mid = float(np.sum(energy[mid_mask]))
    high = float(np.sum(energy[high_mask]))

    # 4x4 區塊內的局部頻帶能量：每格做 8x8 DCT，帶內求和，再跨格統計 mean/max
    grid_h, grid_w = 4, 4
    cell_h = h8 // grid_h
    cell_w = w8 // grid_w
    band_vals = {"low": [], "mid": [], "high": []}
    for gy in range(grid_h):
        for gx in range(grid_w):
            cell = patch[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            cell_blocks = cell.reshape(cell_h // 8, 8, cell_w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
            c_energy = []
            for blk in cell_blocks:
                coeff = dct(dct(blk.T, norm="ortho").T, norm="ortho")
                c_energy.append(np.abs(coeff) ** 2)
            c_energy = np.mean(c_energy, axis=0)
            band_vals["low"].append(float(np.sum(c_energy[low_mask])))
            band_vals["mid"].append(float(np.sum(c_energy[mid_mask])))
            band_vals["high"].append(float(np.sum(c_energy[high_mask])))

    low_mean, low_max = float(np.mean(band_vals["low"])), float(np.max(band_vals["low"]))
    mid_mean, mid_max = float(np.mean(band_vals["mid"])), float(np.max(band_vals["mid"]))
    high_mean, high_max = float(np.mean(band_vals["high"])), float(np.max(band_vals["high"]))

    return low, mid, high, low_mean, low_max, mid_mean, mid_max, high_mean, high_max


def patch_features(residual: np.ndarray) -> np.ndarray:
    r = residual.astype(np.float32)
    mean_r = float(r.mean())
    std_r = float(r.std() + 1e-6)
    # 標準化並 clip 掉尾巴，讓 skew/kurt 更穩定
    r_norm = (r - mean_r) / std_r
    r_norm = np.clip(r_norm, -3.0, 3.0)

    mean_rn = float(r_norm.mean())
    std_rn = float(r_norm.std() + 1e-6)
    skew = float(stats.skew(r_norm.reshape(-1)))
    kurt = float(stats.kurtosis(r_norm.reshape(-1)))
    energy_r = float(np.mean(r_norm ** 2))

    gx = ndimage.sobel(r_norm, axis=1)
    gy = ndimage.sobel(r_norm, axis=0)
    mag = np.hypot(gx, gy)
    mean_grad = float(mag.mean())
    std_grad = float(mag.std() + 1e-6)

    (
        dct_low,
        dct_mid,
        dct_high,
        low_mean,
        low_max,
        mid_mean,
        mid_max,
        high_mean,
        high_max,
    ) = block_dct_energy(r_norm)

    return np.array(
        [
            mean_rn,
            std_rn,
            skew,
            kurt,
            energy_r,
            mean_grad,
            std_grad,
            dct_low,
            dct_mid,
            dct_high,
            low_mean,
            low_max,
            mid_mean,
            mid_max,
            high_mean,
            high_max,
        ],
        dtype=np.float32,
    )


def relation_features(face_feat: np.ndarray, bg_feat: np.ndarray) -> np.ndarray:
    diff = face_feat - bg_feat
    abs_diff = np.abs(diff)
    ratio = face_feat / (bg_feat + 1e-6)
    ratio = np.clip(ratio, 0.1, 10.0)  # winsorize 避免極端值
    return np.concatenate([diff, abs_diff, ratio])


def preprocess_frame(frame: np.ndarray, frame_id: int, label: int) -> PatchSample:
    gray = to_gray(frame)
    face_center, face_size = detect_face_bbox(gray)
    # 擴展倍率調整為 1.0，並限制最大裁剪不超過影像短邊的 80%
    scale = 1.0
    max_side = int(0.8 * min(gray.shape))
    side = min(int(scale * face_size), max_side)
    face_patch = crop_square(gray, face_center, side)
    bg_patch = select_background_patch(gray, face_center, side)
    face_patch = resize_to(face_patch, 128)
    bg_patch = resize_to(bg_patch, 128)
    face_res = compute_residual(face_patch)
    bg_res = compute_residual(bg_patch)
    return PatchSample(frame_id=frame_id, face_patch=face_res, bg_patch=bg_res, label=label)


def build_feature_vector(sample: PatchSample) -> np.ndarray:
    face_feat = patch_features(sample.face_patch)
    bg_feat = patch_features(sample.bg_patch)
    rel = relation_features(face_feat, bg_feat)
    return np.concatenate([face_feat, bg_feat, rel])


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


def process_folder(
    input_dir: str,
    output_dir: str,
    max_images: int = 3,
    label: int = 0,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> List[Tuple[str, int]]:
    """讀取資料夾中的影像（遞迴搜尋 exts），輸出 residual 可視化與特徵向量。"""
    all_paths = []
    for ext in exts:
        all_paths.extend(Path(input_dir).rglob(f"*{ext}"))
    paths = sorted(all_paths)[:max_images]
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, p in enumerate(paths):
        frame = load_image_gray(str(p))
        sample = preprocess_frame(frame, frame_id=i, label=label)
        vec = build_feature_vector(sample)
        save_residual_png(sample.face_patch, os.path.join(output_dir, f"{p.stem}_face_res.png"))
        save_residual_png(sample.bg_patch, os.path.join(output_dir, f"{p.stem}_bg_res.png"))
        np.save(os.path.join(output_dir, f"{p.stem}_feat.npy"), vec)
        results.append((p.name, vec.shape[0]))
    return results


def process_paths(paths: List[Path], output_dir: str, label: int = 0, prefix: str = "") -> List[Tuple[str, int]]:
    """針對指定的影像路徑列表做處理，並在檔名中帶入路徑資訊避免重名覆寫。"""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    def build_name(p: Path) -> str:
        parent = p.parent.name
        grand = p.parent.parent.name if p.parent.parent else ""
        tag = "__".join([x for x in [grand, parent, p.stem] if x])
        return f"{prefix}{tag}"

    for i, p in enumerate(paths):
        frame = load_image_gray(str(p))
        sample = preprocess_frame(frame, frame_id=i, label=label)
        vec = build_feature_vector(sample)
        base = build_name(p)
        save_residual_png(sample.face_patch, os.path.join(output_dir, f"{base}_face_res.png"))
        save_residual_png(sample.bg_patch, os.path.join(output_dir, f"{base}_bg_res.png"))
        np.save(os.path.join(output_dir, f"{base}_feat.npy"), vec)
        results.append((base, vec.shape[0]))
    return results


def select_samples_by_label(
    base_dir: str,
    num_real: int = 1,
    num_fake: int = 1,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> Tuple[List[Path], List[Path]]:
    """
    在 base_dir 下遞迴搜尋影像，按 FF++ 命名慣例判斷 real/fake。
    判定優先看路徑片段（不區分大小寫），常見 fake 模式：deepfakes/df、face2face/f2f、faceswap/fs、neuraltextures/nt。
    若路徑含 real 且不含任一 fake token，視為 real。
    """

    def classify(path: Path) -> str:
        lower = path.as_posix().lower()
        parts = [p.lower() for p in path.parts]
        fake_tokens = ["deepfakes", "df", "face2face", "f2f", "faceswap", "fs", "neuraltextures", "nt"]
        is_fake = any(tok in lower for tok in fake_tokens) or any(tok in parts for tok in fake_tokens)
        is_real = ("real" in lower or "authentic" in lower or "origin" in lower) and not is_fake
        if is_fake:
            return "fake"
        if is_real:
            return "real"
        return "unknown"

    real, fake = [], []
    all_real, all_fake = [], []
    for ext in exts:
        for p in Path(base_dir).rglob(f"*{ext}"):
            cls = classify(p)
            if cls == "fake":
                all_fake.append(p)
            elif cls == "real":
                all_real.append(p)
    if all_real:
        real = random.sample(all_real, min(num_real, len(all_real)))
    if all_fake:
        fake = random.sample(all_fake, min(num_fake, len(all_fake)))
    return real, fake


def visualize_residuals(paths: List[Path], labels: List[str], output_path: str) -> None:
    """顯示 face/bg 殘差對比，保存成圖檔。"""
    n = len(paths)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, (p, lbl) in enumerate(zip(paths, labels)):
        frame = load_image_gray(str(p))
        sample = preprocess_frame(frame, frame_id=i, label=0 if lbl.lower() == "real" else 1)
        axes[0, i].imshow(sample.face_patch, cmap="gray")
        axes[0, i].set_title(f"{lbl} face\n{p.parent.name}/{p.name}", fontsize=8)
        axes[1, i].imshow(sample.bg_patch, cmap="gray")
        axes[1, i].set_title("bg residual", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def run_demo(num_frames: int = 3) -> None:
    samples: List[PatchSample] = []
    for i in range(num_frames):
        frame, _, _ = synthetic_frame()
        label = i % 2  # 假裝交替真/假
        sample = preprocess_frame(frame, frame_id=i, label=label)
        samples.append(sample)

    for sample in samples:
        vec = build_feature_vector(sample)
        print(f"frame {sample.frame_id} label={sample.label} feature_dim={vec.shape[0]}")
        print(f"  face residual mean/std: {sample.face_patch.mean():.3f}/{sample.face_patch.std():.3f}")
        print(f"  bg   residual mean/std: {sample.bg_patch.mean():.3f}/{sample.bg_patch.std():.3f}")
        print(f"  first 5 dims: {vec[:5]}")


if __name__ == "__main__":
    run_demo(num_frames=2)
    # 直接在 /ssd2/deepfake/c23/val 抽樣 1 張 real、1 張 fake（假設環境依賴已就緒）
    sample_dir = "/ssd2/deepfake/c23/val"
    out_dir = "demo_outputs"
    if Path(sample_dir).exists():
        real_paths, fake_paths = select_samples_by_label(sample_dir, num_real=3, num_fake=3)
        print("selected real paths:", real_paths)
        print("selected fake paths:", fake_paths)
        if real_paths:
            res_real = process_paths(real_paths[:1], out_dir, label=0, prefix="real_")
            print(f"real sample saved to {out_dir}: {res_real}")
        if fake_paths:
            res_fake = process_paths(fake_paths[:1], out_dir, label=1, prefix="fake_")
            print(f"fake sample saved to {out_dir}: {res_fake}")
        # 合併前 3 張 real 和前 3 張 fake 做對比視覺化
        viz_paths = real_paths[:3] + fake_paths[:3]
        viz_labels = ["real"] * len(real_paths[:3]) + ["fake"] * len(fake_paths[:3])
        if viz_paths:
            viz_out = os.path.join(out_dir, "residual_compare.png")
            visualize_residuals(viz_paths, viz_labels, viz_out)
            print(f"saved comparison plot to {viz_out}")
        if not real_paths and not fake_paths:
            print("No real/fake samples found under", sample_dir)
