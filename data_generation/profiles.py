"""Commanded force profiles; identical to demo_pybullet_digit_force.py.

Used only as a fallback label when a run has no label CSV.
"""
from __future__ import annotations

import math

import numpy as np


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def profile_shape(kind: str, u: float) -> float:
    """Normalized profile value in [0, 1] for u in [0, 1]."""
    if kind == "square":
        r = 0.03
        x = float(np.clip(min(u / r, (1.0 - u) / r), 0.0, 1.0))
        return 0.5 - 0.5 * math.cos(math.pi * x)
    if kind == "trapezoidal":
        return float(np.clip(min(u / 0.25, (1.0 - u) / 0.25), 0.0, 1.0))
    if kind == "triangular":
        return float(1.0 - abs(2.0 * u - 1.0))
    if kind == "sinusoidal":
        return float(math.sin(math.pi * u))
    if kind == "sigmoid":
        k = 20.0
        raw = _sig(k * (u - 0.2)) - _sig(k * (u - 0.8))
        lo = _sig(k * -0.2) - _sig(k * -0.8)
        hi = _sig(k * 0.3) - _sig(k * -0.3)
        return float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))
    raise ValueError(f"Unknown profile kind: {kind}")


def target_force(kind: str, peak: float, duration: float, t: np.ndarray,
                 pre_rest: float = 0.2, post_rest: float = 0.2) -> np.ndarray:
    t0, t1 = pre_rest * duration, (1.0 - post_rest) * duration
    out = np.zeros_like(t, dtype=float)
    active = (t > t0) & (t < t1)
    for i in np.flatnonzero(active):
        out[i] = peak * profile_shape(kind, (t[i] - t0) / (t1 - t0))
    return out
