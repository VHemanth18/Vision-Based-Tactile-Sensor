"""Evaluation figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def parity_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    y, p = df["force"].to_numpy(), df["pred"].to_numpy()
    lim = max(y.max(), p.max()) * 1.05
    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    hb = ax.hexbin(y, p, gridsize=60, bins="log", mincnt=1, cmap="viridis", extent=(0, lim, 0, lim))
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="ideal")
    ax.set_xlabel("Measured force (N)")
    ax.set_ylabel("Predicted force (N)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.colorbar(hb, ax=ax, label="frames (log)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def error_by_force(df: pd.DataFrame, path: Path) -> None:
    bins = np.linspace(0, df["force"].max() + 1e-6, 21)
    centers = 0.5 * (bins[1:] + bins[:-1])
    which = np.digitize(df["force"], bins) - 1
    err = (df["pred"] - df["force"]).to_numpy()
    mae = [np.mean(np.abs(err[which == i])) if np.any(which == i) else np.nan for i in range(len(centers))]
    bias = [np.mean(err[which == i]) if np.any(which == i) else np.nan for i in range(len(centers))]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    ax.plot(centers, mae, "o-", label="MAE")
    ax.plot(centers, bias, "s--", label="bias (pred - true)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Measured force (N)")
    ax.set_ylabel("Error (N)")
    ax.set_title("Error vs force level")
    ax.grid(True)
    ax.legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_timeseries(run_df: pd.DataFrame, path: Path) -> None:
    """Same style as the data-collection force plot, with the model prediction added."""
    r = run_df.sort_values("time_s")
    first = r.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    ax.plot(r["time_s"], r["target_force"], "--", lw=2, label="Target Force (N)")
    ax.plot(r["time_s"], r["force"], lw=1.5, label="Measured Force (N)")
    ax.plot(r["time_s"], r["pred"], lw=1.2, alpha=0.9, label="Predicted Force (N)")
    ax.axhline(first["peak_force"], ls=":", lw=1, alpha=0.6, label=f"{first['peak_force']:g}N limit")
    mae = np.mean(np.abs(r["pred"] - r["force"]))
    shape = str(first["shape"]).replace("_", " ").title()
    ax.set_title(f"{shape}: {first['peak_force']:g}N, {first['duration']:g}s, "
                 f"{str(first['profile']).capitalize()}  (MAE {mae:.3f} N)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.grid(True)
    ax.legend(loc="center right")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def training_curves(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    ax[0].plot(history["epoch"], history["train_loss"], label="train")
    ax[0].plot(history["epoch"], history["val_loss"], label="val")
    ax[0].set_title("Loss (Huber, scaled)")
    ax[1].plot(history["epoch"], history["val_mae_N"], "o-")
    ax[1].set_title("Validation MAE (N)")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(True)
    ax[0].legend()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
