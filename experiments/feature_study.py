"""Which image statistics carry force information that transfers across indenter shapes?

Extracts hand-crafted statistics of the (denoised) frame-minus-reference difference for a subsample
of frames, then runs leave-one-shape-out (LOSO) regression over the NON-test shapes only
(hexagonal is never used here), so the conclusions do not peek at the test shape.

    python experiments/feature_study.py --index outputs/split_test-hexagonal_val-hash/index/index.csv \
        --out outputs/analysis/features.parquet
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TEST_SHAPE = "hexagonal_indenter"
THR = (0.02, 0.04, 0.06, 0.08, 0.10, 0.14)


def _rd(p: str) -> np.ndarray:
    return cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0


_REF: dict[str, np.ndarray] = {}


def feats(args) -> dict:
    path, ref_path, sigma = args
    ref = _REF.get(ref_path)
    if ref is None:
        ref = _REF[ref_path] = _rd(ref_path)
    d = cv2.GaussianBlur(_rd(path) - ref, (0, 0), sigma)
    n = np.sqrt((d ** 2).sum(2))
    out = {f"p{q}": float(np.percentile(n, q)) for q in (50, 90, 95, 99, 99.9)}
    out["max"] = float(n.max())
    out["sum"] = float(n.sum())
    for t in THR:
        out[f"area{t}"] = float((n > t).mean())
    m = n > 0.04
    for c, name in enumerate("rgb"):
        out[f"mean_{name}"] = float(d[..., c][m].mean()) if m.any() else 0.0
    # intensity-free geometry: ratios of areas at different thresholds (edge sharpness / depth proxy)
    a_lo = out["area0.02"] + 1e-6
    for t in THR[1:]:
        out[f"ratio{t}"] = out[f"area{t}"] / a_lo
    out["p99_over_p90"] = out["p99"] / (out["p90"] + 1e-6)
    out["sum_over_area"] = out["sum"] / (a_lo * n.size)
    return out


def extract(index_csv: str, out: str, stride: int, sigma: float, workers: int) -> pd.DataFrame:
    idx = pd.read_csv(index_csv)
    idx = idx.sort_values(["run_id", "frame_idx"])
    idx = idx[idx.groupby("run_id").cumcount() % stride == 0]
    idx = idx.sort_values("ref_path").reset_index(drop=True)  # keeps a worker's reference cache warm
    print(f"extracting {len(idx)} frames ...", flush=True)
    with Pool(workers) as pool:
        rows = pool.map(feats, [(p, r, sigma) for p, r in zip(idx["path"], idx["ref_path"])], chunksize=256)
    df = pd.concat([idx[["run_id", "shape", "profile", "peak_force", "duration", "frame_idx", "force"]],
                    pd.DataFrame(rows)], axis=1)
    df.to_parquet(out) if out.endswith(".parquet") else df.to_csv(out, index=False)
    return df


FEATURE_SETS = {
    "intensity (percentiles,max,sum)": ["p50", "p90", "p95", "p99", "p99.9", "max", "sum"],
    "areas": [f"area{t}" for t in THR],
    "ratios only (intensity-free)": [f"ratio{t}" for t in THR[1:]] + ["p99_over_p90", "sum_over_area"],
    "all": None,
}


def loso(df: pd.DataFrame) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = df[df["shape"] != TEST_SHAPE]
    feat_all = [c for c in df.columns if c not in
                ("run_id", "shape", "profile", "peak_force", "duration", "frame_idx", "force")]
    shapes = sorted(df["shape"].unique())
    res = {}
    for name, cols in FEATURE_SETS.items():
        cols = cols or feat_all
        per = {}
        for s in shapes:
            tr, te = df[df["shape"] != s], df[df["shape"] == s]
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
            m.fit(tr[cols], tr["force"])
            p = np.clip(m.predict(te[cols]), 0, None)
            c = te["force"].to_numpy() > 0.05
            per[s.replace("_indenter", "")] = float(np.mean(np.abs(p[c] - te["force"].to_numpy()[c])))
        per["MEAN"] = float(np.mean(list(per.values())))
        res[name] = per
    print("\nLOSO in-contact MAE (N), hexagonal excluded:")
    print(pd.DataFrame(res).T.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    if a.reuse and Path(a.out).exists():
        df = pd.read_parquet(a.out) if a.out.endswith(".parquet") else pd.read_csv(a.out)
    else:
        df = extract(a.index, a.out, a.stride, a.sigma, a.workers)
    loso(df)
    sys.exit(0)
