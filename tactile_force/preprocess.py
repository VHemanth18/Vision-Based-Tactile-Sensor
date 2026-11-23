"""Image loading and preprocessing shared by training, evaluation and prediction."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RGB_MEAN = np.float32(0.5)  # simple fixed normalisation of [0, 1] pixels
RGB_STD = np.float32(0.25)


def read_rgb(path: str | Path) -> np.ndarray:
    """uint8 HxWx3 RGB. np.fromfile + imdecode handles non-ASCII Windows paths."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_and_resize(img: np.ndarray, pp) -> np.ndarray:
    """Mask the text band, crop the configured panel, resize. Returns float32 HxWx3 in [0, 1]."""
    img = img.copy()
    if pp.mask_top_px:
        img[: int(pp.mask_top_px)] = 0
    h, w = img.shape[:2]
    x0, x1, y0, y1 = pp.crop
    img = img[int(round(y0 * h)): int(round(y1 * h)), int(round(x0 * w)): int(round(x1 * w))]
    out_h, out_w = pp.size
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_pair(path: str, ref_path: str | None, pp) -> tuple[np.ndarray, np.ndarray | None]:
    img = crop_and_resize(read_rgb(path), pp)
    ref = crop_and_resize(read_rgb(ref_path), pp) if ref_path else None
    return img, ref


def augment_pair(img: np.ndarray, ref: np.ndarray | None, rng: np.random.Generator):
    """Geometric + photometric jitter applied identically to frame and reference."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
        ref = ref[:, ::-1] if ref is not None else None
    if rng.random() < 0.5:
        img = img[::-1]
        ref = ref[::-1] if ref is not None else None
    dy, dx = rng.integers(-4, 5, size=2)
    img = np.roll(img, (dy, dx), axis=(0, 1))
    ref = np.roll(ref, (dy, dx), axis=(0, 1)) if ref is not None else None
    gain = rng.uniform(0.9, 1.1)
    bias = rng.uniform(-0.03, 0.03)
    img = np.clip(img * gain + bias, 0, 1)
    ref = np.clip(ref * gain + bias, 0, 1) if ref is not None else None
    img = np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)
    return np.ascontiguousarray(img), (np.ascontiguousarray(ref) if ref is not None else None)


def to_input(img: np.ndarray, ref: np.ndarray | None, pp) -> np.ndarray:
    """Build the CHW float32 network input for the configured input_mode."""
    mode = pp.input_mode
    raw = ((img - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
    if mode == "raw":
        return raw.astype(np.float32)
    if ref is None:
        raise ValueError(f"input_mode={mode} needs a reference (no-contact) frame")
    diff = ((img - ref) * pp.diff_gain).transpose(2, 0, 1)
    if mode == "diff":
        return diff.astype(np.float32)
    if mode == "raw+diff":
        return np.concatenate([raw, diff], axis=0).astype(np.float32)
    raise ValueError(f"Unknown input_mode: {mode}")


def in_channels(pp) -> int:
    return (6 if pp.input_mode == "raw+diff" else 3) + 3 * len(pp.get("temporal_offsets") or [])
