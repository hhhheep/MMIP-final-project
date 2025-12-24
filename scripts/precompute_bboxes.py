#!/usr/bin/env python3
"""
Standalone bbox warmup:
 - 讀取 manifest JSONL（split/method/video_id/frames_root）。
 - 依 rPPG meta 的時間窗或全長建立取樣計畫（或直接使用已存在的 plan）。
 - 對計畫涵蓋的幀生成/整理人臉 bbox，寫入 FaceTrackCache。

用法範例：
python scripts/precompute_bboxes.py \
  --manifest manifest.jsonl \
  --bbox-cache bbox_cache \
  --rppg-cache runs/rppg_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.mm.face_track_cache import FaceTrackCache, FaceTrackConfig  # noqa: E402
from common.mm.sample_plan import make_sample_plan  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="manifest JSONL (split/method/video_id/frames_root).")
    ap.add_argument("--bbox-cache", type=Path, default=Path("bbox_cache"))
    ap.add_argument("--rppg-cache", type=Path, default=Path("runs/rppg_cache"), help="rPPG cache/meta 根目錄。")
    ap.add_argument("--plan-root", type=Path, default=None, help="若提供且存在對應檔，優先讀取既有 plan。")
    ap.add_argument("--frames-per-video", type=int, default=8, help="計畫中要取的幀數 T。")
    ap.add_argument("--strategy", type=str, default="uniform", choices=["uniform", "random", "fixed"])
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def read_manifest(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def video_uid(split: str, method: str, video_id: str) -> str:
    return hashlib.sha1(f"{split}|{method}|{video_id}".encode("utf-8")).hexdigest()


def load_rppg_meta(cache_root: Path, split: str, method: str, video_id: str) -> Dict:
    meta_dir = cache_root / split / method / video_id
    meta_files = sorted(meta_dir.glob("*.meta.json"))
    if not meta_files:
        return {}
    with meta_files[-1].open("r") as f:
        return json.load(f)


def list_frames(frames_dir: Path) -> List[Path]:
    list_file = frames_dir / "list.txt"
    if list_file.exists():
        ordered: List[Path] = []
        with list_file.open("r") as f:
            for line in f:
                name = line.strip()
                if name:
                    p = frames_dir / name
                    if p.exists():
                        ordered.append(p)
        if ordered:
            return ordered
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        frames = sorted(frames_dir.glob(pattern))
        if frames:
            return frames
    return []


def load_plan(plan_root: Optional[Path], vid_uid: str) -> Optional[List[int]]:
    if not plan_root:
        return None
    path = plan_root / f"{vid_uid}.json"
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "frame_indices" in data:
            return list(data["frame_indices"])
    except Exception:
        return None
    return None


def main():
    args = parse_args()
    manifest = read_manifest(args.manifest)
    cache = FaceTrackCache(cache_root=args.bbox_cache, config=FaceTrackConfig())

    stats = {
        "videos_total": 0,
        "frames_selected": 0,
        "frames_bbox_filled": 0,
        "skipped_no_frames": 0,
        "skipped_meta_window_invalid": 0,
    }

    for rec in manifest:
        split = rec["split"]
        method = rec["method"]
        video_id = rec["video_id"]
        vid_uid = rec.get("video_uid") or video_uid(split, method, video_id)
        frames_root = Path(rec["frames_root"])
        frames_dir = frames_root / split / method / video_id
        frames = list_frames(frames_dir)
        if not frames:
            stats["skipped_no_frames"] += 1
            continue

        meta = load_rppg_meta(args.rppg_cache, split, method, video_id)
        plan_indices = load_plan(args.plan_root, vid_uid)
        if plan_indices is None:
            window_start = int(meta.get("window_start_frame", 0))
            window_end = int(meta.get("window_end_frame", len(frames)))
            frame_list_len = int(meta.get("frame_list_len", len(frames)))
            if window_end <= window_start:
                window_end = window_start + frame_list_len
            if window_end <= window_start:
                stats["skipped_meta_window_invalid"] += 1
                continue
            plan = make_sample_plan(
                video_uid=vid_uid,
                window_start=window_start,
                window_end=window_end,
                full_len=frame_list_len,
                T=args.frames_per_video,
                strategy=args.strategy,
                seed=args.seed,
                frame_index_basis=meta.get("frame_index_basis", "list_order"),
            )
            plan_indices = plan.frame_indices

        chosen_paths: List[Path] = []
        for idx in plan_indices:
            if 0 <= idx < len(frames):
                chosen_paths.append(frames[idx])
        if not chosen_paths:
            stats["skipped_meta_window_invalid"] += 1
            continue

        stats["videos_total"] += 1
        stats["frames_selected"] += len(chosen_paths)

        result = cache.ensure(vid_uid, chosen_paths)
        stats["frames_bbox_filled"] += sum(1 for v in result.values() if v is not None)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
