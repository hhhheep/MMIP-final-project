"""
Sample plan generator: aligns frame sampling to a given window (e.g., from rPPG meta).

Designed to be pluggable:
- Provide window_start/end (frame indices) and full frame list.
- Choose strategy (uniform/fixed/random) and seed to make sampling reproducible.
- Emits a plan dict with frame_indices and hashes for traceability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class SamplePlan:
    video_uid: str
    window_start: int
    window_end: int  # end-exclusive
    strategy: str
    seed: int
    frame_indices: List[int]
    frame_index_basis: str = "list_order"
    plan_hash: str = ""

    def to_dict(self) -> Dict:
        data = asdict(self)
        if not self.plan_hash:
            data["plan_hash"] = _hash_plan(data)
        return data


def _hash_plan(data: Dict) -> str:
    blob = json.dumps(
        {
            "video_uid": data["video_uid"],
            "window": [data["window_start"], data["window_end"]],
            "strategy": data["strategy"],
            "seed": data["seed"],
            "frame_indices": data["frame_indices"],
            "frame_index_basis": data.get("frame_index_basis", "list_order"),
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def make_sample_plan(
    video_uid: str,
    window_start: int,
    window_end: int,
    full_len: int,
    T: int,
    strategy: str = "uniform",
    seed: int = 1337,
    frame_index_basis: str = "list_order",
) -> SamplePlan:
    """
    Generate a SamplePlan within [window_start, window_end) over a sequence of length full_len.
    """
    window_len = max(0, min(full_len, window_end) - window_start)
    if window_len <= 0:
        raise ValueError("Invalid window length.")

    rng = np.random.default_rng(seed)
    if T is None or T >= window_len:
        idxs_rel = np.arange(window_len, dtype=int)
    else:
        if strategy == "random":
            idxs_rel = np.sort(rng.choice(window_len, size=T, replace=False))
        elif strategy == "fixed":
            # deterministic per video_uid + seed by using RNG but fixed seed
            idxs_rel = np.sort(rng.choice(window_len, size=T, replace=False))
        else:  # uniform
            idxs_rel = np.linspace(0, window_len - 1, T, dtype=int)

    frame_indices = (idxs_rel + window_start).tolist()
    plan = SamplePlan(
        video_uid=video_uid,
        window_start=window_start,
        window_end=window_end,
        strategy=strategy,
        seed=seed,
        frame_indices=frame_indices,
        frame_index_basis=frame_index_basis,
    )
    plan.plan_hash = _hash_plan(plan.to_dict())
    return plan
