"""Image statistics vs measured force, per shape (from feature_study.py output)."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

df = pd.read_parquet(sys.argv[1])
out = sys.argv[2]
cols = ["p99", "p99.9", "sum", "area0.04", "area0.1", "ratio0.1"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=90)
for ax, c in zip(axes.ravel(), cols):
    for s, g in df[df["profile"] == "triangular"].groupby("shape"):
        g = g[g["force"] > 0.02]
        ax.scatter(g["force"], g[c], s=3, label=s.replace("_indenter", ""),
                   color="black" if s.startswith("hexagonal") else None, zorder=3 if s.startswith("hex") else 1)
    ax.set_title(c)
    ax.set_xlabel("measured force (N)")
axes[0, 0].legend(markerscale=4)
fig.tight_layout()
fig.savefig(out)
print("wrote", out)
