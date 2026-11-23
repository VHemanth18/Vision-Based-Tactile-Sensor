"""Post-hoc temporal smoothing of per-frame predictions within each run.
The window is chosen on the experiment's VALIDATION runs (calibration-val runs), then applied once to the test
predictions. Writes <exp>/smoothing/{val_search.csv,test_metrics.json,predictions_smoothed.csv}.
    python experiments/smooth_study.py outputs/experiments/exp007_...
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tactile_force import pipeline
from tactile_force.config import load_config
from tactile_force.metrics import full_report
from tactile_force.train import load_model, make_loader, pick_device, predict_loader


def smooth(df, kind, w):
    if w <= 1:
        return df["pred"].to_numpy()
    out = np.empty(len(df))
    for _, idx in df.groupby("run_id", sort=False).indices.items():
        s = df["pred"].iloc[idx]
        order = np.argsort(df["frame_idx"].to_numpy()[idx]); s = s.iloc[order]
        center = kind.startswith("centered")
        r = s.rolling(w, center=center, min_periods=1)
        v = (r.median() if kind.endswith("median") else r.mean()).to_numpy()
        out[np.asarray(idx)[order]] = v
    return out


def main(exp: Path):
    cfg = load_config([exp / "config_used.yaml"], [])
    parts = pipeline._splits(cfg)
    dev = pick_device(cfg.train.device)
    val = parts["val"].reset_index(drop=True)
    model = load_model(exp / "best_model.pt", cfg, dev)
    val["pred"] = predict_loader(model, make_loader(val, cfg, train=False, batch_size=256), dev, cfg.train.force_scale)
    th = cfg.evaluate.contact_threshold
    rows = []
    for kind in ["causal_mean", "causal_median", "centered_mean", "centered_median"]:
        for w in [1, 3, 5, 7, 9, 13, 17, 25]:
            v = val.copy(); v["pred"] = smooth(val, kind, w)
            r = full_report(v, th)
            rows.append({"kind": kind, "w": w, "val_contact_mae": r["in_contact"]["mae_N"], "val_all_mae": r["overall"]["mae_N"]})
    srch = pd.DataFrame(rows)
    out = exp / "smoothing"; out.mkdir(exist_ok=False)
    srch.to_csv(out / "val_search.csv", index=False)
    print(srch.to_string(index=False))
    test = pd.read_csv(exp / "eval_test" / "predictions.csv")
    res = {"raw": full_report(test, th)}
    for kind in ["causal_mean", "causal_median", "centered_mean", "centered_median"]:
        b = srch[srch.kind == kind].sort_values("val_contact_mae").iloc[0]
        t = test.copy(); t["pred"] = smooth(test, kind, int(b.w))
        res[f"{kind}_w{int(b.w)}"] = full_report(t, th)
        if kind == "causal_mean": t.to_csv(out / "predictions_causal_smoothed.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(res, indent=2))
    for k, r in res.items():
        print(f"TEST {k:22s} contact MAE {r['in_contact']['mae_N']:.4f}  RMSE {r['in_contact']['rmse_N']:.4f}  "
              f"R2 {r['in_contact']['r2']:.4f} | all MAE {r['overall']['mae_N']:.4f} R2 {r['overall']['r2']:.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
