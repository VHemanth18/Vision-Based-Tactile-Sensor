"""Metrics of the shape classifier from its saved test predictions (preds_test.parquet).

    python experiments/shape_classify_eval.py outputs/experiments/exp024_cnn_shape_classification
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

d = Path(sys.argv[1])
te = pd.read_parquet(d / "preds_test.parquet")
thr = 0.05
classes = sorted(te["shape"].unique())
ok = te["pred"] == te["shape"]
# run level: majority vote over the in-contact frames of the run
vote = te.groupby("run_id")["pred"].agg(lambda s: s.value_counts().idxmax())
truth = te.groupby("run_id")["shape"].first()
bins = [thr, 0.25, 0.5, 1, 2, 3, 5.5]
te["fbin"] = pd.cut(te["force"], bins)
by_force = te.assign(ok=ok).groupby("fbin", observed=True)["ok"].agg(n="size", acc="mean")
by_shape = te.assign(ok=ok).groupby("shape")["ok"].mean()
cm = pd.crosstab(te["shape"], te["pred"]).reindex(index=classes, columns=classes, fill_value=0)
cm.to_csv(d / "confusion_frames.csv"); by_force.to_csv(d / "acc_by_force.csv"); by_shape.to_csv(d / "acc_by_shape.csv")
res = {"frame_acc": float(ok.mean()), "macro_f1": float(f1_score(te["shape"], te["pred"], average="macro")),
       "run_acc_majority": float((vote == truth.loc[vote.index]).mean()), "n_test_frames": int(len(te)),
       "n_test_runs": int(len(truth)), "mean_conf_correct": float(te.loc[ok, "conf"].mean()),
       "mean_conf_wrong": float(te.loc[~ok, "conf"].mean()) if (~ok).any() else None}
json.dump(res, open(d / "test_metrics.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print(by_force); print(by_shape); print(cm)
