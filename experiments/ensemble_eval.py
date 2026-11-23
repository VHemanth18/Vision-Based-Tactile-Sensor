"""Evaluate test-time flip augmentation (TTA), seed ensembles and temporal smoothing on top of trained experiments.
All experiments must share one split. Any choice (TTA on/off, smoothing window) is made on the VALIDATION runs only.
    python experiments/ensemble_eval.py --out <dir> --exps outputs/experiments/exp014_... outputs/experiments/exp015_...
Per-model predictions are cached in <exp>/preds_{val,test}_{plain,tta}.parquet (never overwritten).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tactile_force import pipeline
from tactile_force.config import load_config
from tactile_force.metrics import full_report
from tactile_force.train import load_model, make_loader, pick_device
sys.path.insert(0, str(ROOT / "experiments"))
from smooth_study import smooth  # noqa: E402

FLIPS = [(), (3,), (2,), (2, 3)]


@torch.no_grad()
def predict(model, loader, dev, scale, tta):
    out = []
    for x, _ in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p = torch.stack([model(torch.flip(x, f) if f else x).float() for f in (FLIPS if tta else [()])]).mean(0)
        out.append(p.cpu().numpy() * scale)
    return np.clip(np.concatenate(out), 0, None)


def preds_for(exp: Path, part: str, tta: bool):
    f = exp / f"preds_{part}_{'tta' if tta else 'plain'}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cfg = load_config([exp / "config_used.yaml"], [])
    df = pipeline._splits(cfg)[part].reset_index(drop=True)
    dev = pick_device(cfg.train.device)
    m = load_model(exp / "best_model.pt", cfg, dev)
    df["pred"] = predict(m, make_loader(df, cfg, train=False, batch_size=256), dev, float(cfg.train.force_scale), tta)
    df.to_parquet(f)
    return df


def combine(exps, part, tta):
    dfs = [preds_for(e, part, tta) for e in exps]
    base = dfs[0].copy()
    for d in dfs[1:]:
        assert (d["path"].to_numpy() == base["path"].to_numpy()).all(), "experiments do not share a split"
    base["pred"] = np.mean([d["pred"].to_numpy() for d in dfs], axis=0)
    return base


def summ(r):
    return {"contact_mae": r["in_contact"]["mae_N"], "contact_rmse": r["in_contact"]["rmse_N"],
            "contact_bias": r["in_contact"]["bias_N"], "contact_r2": r["in_contact"]["r2"],
            "all_mae": r["overall"]["mae_N"], "all_r2": r["overall"]["r2"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exps", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exps = [Path(e) for e in a.exps]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    th = load_config([exps[0] / "config_used.yaml"], []).evaluate.contact_threshold
    rows = []
    for tta in (False, True):
        v, t = combine(exps, "val", tta), combine(exps, "test", tta)
        for kind in ["none", "causal_mean", "centered_median"]:
            for w in ([1] if kind == "none" else [3, 5, 7, 9, 13, 17, 25, 33, 49, 65]):
                vv, tt = v.copy(), t.copy()
                vv["pred"], tt["pred"] = smooth(v, kind, w), smooth(t, kind, w)
                rows.append({"tta": tta, "smooth": kind, "w": w,
                             **{"val_" + k: x for k, x in summ(full_report(vv, th)).items()},
                             **{"test_" + k: x for k, x in summ(full_report(tt, th)).items()}})
    res = pd.DataFrame(rows); res.to_csv(out / "grid.csv", index=False)
    # selection on validation only: best per smoothing family (none / real-time causal / offline centered)
    chosen = res.loc[res.groupby("smooth")["val_contact_mae"].idxmin()]
    chosen.to_csv(out / "chosen_by_val.csv", index=False)
    (out / "exps.txt").write_text("\n".join(map(str, exps)) + "\n")
    cols = ["tta", "smooth", "w", "val_contact_mae", "test_contact_mae", "test_contact_rmse", "test_contact_bias",
            "test_contact_r2", "test_all_mae", "test_all_r2"]
    print(f"members: {[e.name for e in exps]}")
    print("SELECTED ON VAL\n" + chosen[cols].round(4).to_string(index=False))
    print("no-smoothing rows\n" + res[res.smooth == "none"][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
