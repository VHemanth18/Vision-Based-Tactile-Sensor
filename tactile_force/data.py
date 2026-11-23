"""PyTorch dataset over the frame index."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess import augment_pair, load_pair, to_input


def past_frame_path(path: str, frame_idx: int, k: int) -> str:
    """Path of the frame k steps earlier in the same run (clamped to frame 1); frames are named frame_%05d.png."""
    head, tail = path.rsplit("/", 1)
    ext = tail[tail.rfind("."):]
    return f"{head}/frame_{max(1, int(frame_idx) - k):05d}{ext}"


class FrameForceDataset(Dataset):
    def __init__(self, frames: pd.DataFrame, pp, augment: bool = False, seed: int = 0):
        self.paths = frames["path"].tolist()
        needs_ref = pp.input_mode != "raw"
        self.refs = frames["ref_path"].tolist() if needs_ref else [None] * len(frames)
        self.force = frames["force"].to_numpy(dtype=np.float32)
        self.pp, self.augment, self.seed = pp, augment, seed
        self._ref_cache: dict[str, np.ndarray] = {}
        self._rng: np.random.Generator | None = None
        self.offsets = [int(k) for k in (pp.get("temporal_offsets") or [])]  # e.g. [2, 4]: past frames t-2, t-4
        self.fidx = frames["frame_idx"].to_numpy() if self.offsets else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img, _ = load_pair(self.paths[i], None, self.pp)
        ref = None
        rp = self.refs[i]
        if rp is not None:  # one reference per run; cache per worker
            ref = self._ref_cache.get(rp)
            if ref is None:
                k = int(self.pp.get("ref_frames") or 1)  # average the first k (no-contact) frames: less sensor noise
                ref = np.mean([load_pair(past_frame_path(rp, j + 1, 0), None, self.pp)[0] for j in range(k)],
                              axis=0).astype(np.float32) if k > 1 else load_pair(rp, None, self.pp)[0]
                self._ref_cache[rp] = ref
        past = [load_pair(past_frame_path(self.paths[i], self.fidx[i], k), None, self.pp)[0] for k in self.offsets]
        if past:  # augment frame and past frames identically by stacking them along the channel axis
            img = np.concatenate([img] + past, axis=2)
        if self.augment:
            if self._rng is None:  # one generator per worker process, seeded by torch per epoch
                self._rng = np.random.default_rng((self.seed, torch.initial_seed() % (2**32)))
            img, ref = augment_pair(img, ref, self._rng)
            jit = self.pp.get("diff_gain_jitter")  # e.g. [0.6, 1.6]: random contrast of the deformation
            if jit and ref is not None:
                img = np.ascontiguousarray(ref + self._rng.uniform(*jit) * (img - ref), dtype=np.float32)
        if past:
            img, past = img[..., :3], [img[..., 3 * (j + 1):3 * (j + 2)] for j in range(len(past))]
        x = to_input(img, ref, self.pp)
        if past:  # extra channels: deformation of each past frame, (past - ref) * diff_gain
            x = np.concatenate([x] + [((q - ref) * self.pp.diff_gain).transpose(2, 0, 1).astype(np.float32)
                                      for q in past], axis=0)
        return torch.from_numpy(x), torch.tensor(self.force[i])


def subsample_per_run(frames: pd.DataFrame, max_per_run: int | None, seed: int) -> pd.DataFrame:
    """Random subset of at most max_per_run frames from each run (re-drawn every epoch)."""
    if not max_per_run:
        return frames
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in frames.groupby("run_id").indices.items():
        keep.append(idx if len(idx) <= max_per_run else rng.choice(idx, max_per_run, replace=False))
    return frames.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
