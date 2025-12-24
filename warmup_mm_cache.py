#!/usr/bin/env python3
"""
Warmup utility to precompute sample plans and face bboxes.
Non-destructive: works alongside existing pipelines.

Flow:
 1) Scan a manifest JSONL (split/method/video_id/frames_root required).
 2) For each record, load rPPG meta (if available) to get window_start/end and frame list digest.
 3) Generate SamplePlan (T=8 by default, uniform strategy).
 4) Ensure face bboxes via FaceTrackCache for the frames in plan.

Outputs:
 - plan JSON under runs/mm_plans/{video_uid}.json
 - bbox cache under provided cache root
 - summary stats printed at end
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from common.mm.face_track_cache import FaceTrackCache, FaceTrackConfig
from common.mm.sample_plan import make_sample_plan


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSONL (split/method/video_id/frames_root).")
    ap.add_argument("--plans-out", type=Path, default=Path("runs/mm_plans"))
    ap.add_argument("--bbox-cache", type=Path, default=Path("bbox_cache"))
    ap.add_argument("--T", type=int, default=8, help="Frames per video for RGB/Noise sampling.")
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


def load_rppg_meta(frames_root: Path, record: Dict) -> Dict:
    """
    Locate the rPPG meta next to cached spectrogram using the key from the record.
    Expect path: runs/rppg_cache/<split>/<method>/<video_id>/*.meta.json
    """
    split = record["split"]
    method = record["method"]
    video_id = record["video_id"]
    cache_root = Path(record.get("rppg_cache_root", "runs/rppg_cache"))
    meta_dir = cache_root / split / method / video_id
    meta_files = sorted(meta_dir.glob("*.meta.json"))
    if not meta_files:
        return {}
    # choose the first (or latest by name)
    with meta_files[-1].open("r") as f:
        return json.load(f)


def main():
    args = parse_args()
    plans_out = args.plans_out
    plans_out.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.manifest)

    stats = {
        "total": 0,
        "rppg_meta_missing": 0,
        "plan_written": 0,
        "bbox_missing": 0,
        "bbox_filled": 0,
    }

    cache = FaceTrackCache(cache_root=args.bbox_cache, config=FaceTrackConfig())

    for rec in manifest:
        stats["total"] += 1
        meta = load_rppg_meta(Path(rec["frames_root"]), rec)
        if not meta:
            stats["rppg_meta_missing"] += 1
            continue

        window_start = int(meta.get("window_start_frame", 0))
        window_end = int(meta.get("window_end_frame", window_start))
        frame_list_len = int(meta.get("frame_list_len", window_end - window_start))
        video_uid = meta.get("video_uid") or rec.get("video_uid")
        if video_uid is None:
            video_uid = rec["video_id"]

        plan = make_sample_plan(
            video_uid=video_uid,
            window_start=window_start,
            window_end=window_end,
            full_len=frame_list_len,
            T=args.T,
            strategy=args.strategy,
            seed=args.seed,
            frame_index_basis=meta.get("frame_index_basis", "list_order"),
        )

        plan_path = plans_out / f"{video_uid}.json"
        with plan_path.open("w") as f:
            json.dump(plan.to_dict(), f, indent=2)
        stats["plan_written"] += 1

        # Build frame paths for bbox ensuring.
        frame_root = Path(rec["frames_root"])
        frame_source = meta.get("frame_list_source", "list_order")
        frames_dir = frame_root / rec["split"] / rec["method"] / rec["video_id"]

        # Reconstruct the ordered frame list to align indices.
        frames = []
        list_file = frames_dir / "list.txt"
        if "list.txt" in frame_source and list_file.exists():
            with list_file.open("r") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        frames.append(frames_dir / name)
        else:
            for pattern in ("*.png", "*.jpg", "*.jpeg"):
                frames = sorted(frames_dir.glob(pattern))
                if frames:
                    break

        # Map indices to paths
        chosen_paths = []
        for idx in plan.frame_indices:
            if idx < 0 or idx >= len(frames):
                stats["bbox_missing"] += 1
                continue
            chosen_paths.append(frames[idx])
        # Ensure bboxes
        result = cache.ensure(video_uid, chosen_paths)
        stats["bbox_filled"] += sum(1 for v in result.values() if v is not None)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
