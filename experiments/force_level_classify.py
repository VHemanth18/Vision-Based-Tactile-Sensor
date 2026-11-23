"""Force-level classification derived from the force-regression predictions (no extra training).

Two tasks on the test runs of an experiment:
  * run level  - which of the five peak-force levels (1..5 N) was applied? predicted peak = maximum of the
                 smoothed per-frame prediction of the run, assigned to the nearest level;
  * frame level - which force band is the frame in? classes: no contact (<= 0.05 N), (0.05,0.5], then bands of 1 N centred on 1..5 N.
Smoothing settings are the ones chosen on the validation runs (ensemble_eval.py).

    python experiments/force_level_classify.py outputs/experiments/exp021_cnn_cal5v5_hex_ref30_last
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smooth_study import smooth  # noqa: E402

BANDS = [-np.inf, 0.05, 0.5, 1.5, 2.5, 3.5, 4.5, np.inf]  # bands centred on the commanded levels 1..5 N
NAMES = ["no contact", "0.05-0.5 N", "~1 N", "~2 N", "~3 N", "~4 N", "~5 N"]


def band(x):
    return pd.cut(x, BANDS, labels=False)


def main(exp: str) -> None:
    d = Path(exp)
    df = pd.read_parquet(d / "preds_test_tta.parquet")
    out = d / "force_level_classification"
    out.mkdir(exist_ok=True)
    res = {}
    for name, kind, w in [("single_frame", "none", 1), ("causal_mean_w3", "causal_mean", 3),
                          ("centered_median_w25", "centered_median", 25)]:
        p = smooth(df, kind, w)
        yt, yp = band(df["force"].to_numpy()), band(p)
        cm = pd.crosstab(pd.Series(yt, name="true"), pd.Series(yp, name="pred")).reindex(
            index=range(7), columns=range(7), fill_value=0)
        cm.index, cm.columns = NAMES, NAMES
        cm.to_csv(out / f"frame_confusion_{name}.csv")
        res[f"frame_{name}"] = {"acc": float((yt == yp).mean()), "within_one": float((np.abs(yt - yp) <= 1).mean()),
                                "acc_in_contact": float((yt == yp)[yt > 0].mean())}
    # run level
    p = smooth(df, "centered_median", 25)
    g = df.assign(ps=p).groupby("run_id").agg(true=("peak_force", "first"), peak=("ps", "max"))
    g["pred"] = np.clip(np.rint(g["peak"]), 1, 5)
    g.to_csv(out / "run_level_predictions.csv")
    cm = pd.crosstab(g["true"], g["pred"]).reindex(index=[1.0, 2, 3, 4, 5], columns=[1.0, 2, 3, 4, 5], fill_value=0)
    cm.to_csv(out / "run_confusion.csv")
    res["run_level"] = {"acc": float((g["true"] == g["pred"]).mean()), "n_runs": int(len(g)),
                        "peak_mae_N": float(np.mean(np.abs(g["peak"] - g["true"])))}
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1)); print(cm)


if __name__ == "__main__":
    for e in sys.argv[1:]:
        print("==", e); main(e)
