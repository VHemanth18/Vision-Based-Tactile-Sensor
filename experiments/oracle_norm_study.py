"""Upper-bound check for temporal/per-run normalisation on unseen shapes (hexagonal excluded).

Each frame's statistics are divided by the same run's statistics at ~1 N (an ORACLE: a real sequence model
would have to estimate that reference). If leave-one-shape-out error stays large even with this oracle, the
creep above saturation is not a shape-invariant force cue and no single-run normalisation can fix it.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

df = pd.read_parquet(sys.argv[1])
df = df[df["shape"] != "hexagonal_indenter"]
stats = ["p99", "p99.9", "sum", "area0.04", "area0.1", "p95", "max"]
ref = (df[(df["force"] > 0.9) & (df["force"] < 1.1)].groupby("run_id")[stats].median())
d = df[(df["force"] > 1.0) & df["run_id"].isin(ref.index)].copy()
for c in stats:
    d[c + "_rel"] = d[c] / d["run_id"].map(ref[c])
sets = {"absolute": stats, "relative to own run @1N (oracle)": [c + "_rel" for c in stats]}
out = {}
for name, cols in sets.items():
    per = {}
    for s in sorted(d["shape"].unique()):
        tr, te = d[d["shape"] != s], d[d["shape"] == s]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(tr[cols], tr["force"])
        per[s.replace("_indenter", "")] = float(np.mean(np.abs(np.clip(m.predict(te[cols]), 0, None) - te["force"])))
    per["MEAN"] = float(np.mean(list(per.values())))
    out[name] = per
const = {s.replace("_indenter", ""): float(np.mean(np.abs(d[d["shape"] == s]["force"] - d[d["shape"] != s]["force"].median())))
         for s in sorted(d["shape"].unique())}
const["MEAN"] = float(np.mean(list(const.values())))
out["constant (median of other shapes)"] = const
print("LOSO MAE (N) on frames with force > 1 N:")
print(pd.DataFrame(out).T.round(3).to_string())
