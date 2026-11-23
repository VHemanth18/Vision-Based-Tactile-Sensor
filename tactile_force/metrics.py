"""Regression metrics, overall and per group."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "mae_N": float(np.mean(np.abs(err))),
        "rmse_N": float(np.sqrt(np.mean(err ** 2))),
        "bias_N": float(np.mean(err)),
        "p95_abs_err_N": float(np.percentile(np.abs(err), 95)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def full_report(df: pd.DataFrame, contact_threshold: float) -> dict:
    """df needs columns force, pred, shape, profile, peak_force."""
    y, p = df["force"].to_numpy(), df["pred"].to_numpy()
    contact = y > contact_threshold
    out = {"overall": regression_metrics(y, p)}
    if contact.any():
        out["in_contact"] = regression_metrics(y[contact], p[contact])
        out["in_contact"]["mape_pct"] = float(np.mean(np.abs(p[contact] - y[contact]) / y[contact]) * 100)
    if (~contact).any():
        out["no_contact"] = {"mae_N": float(np.mean(np.abs(p[~contact])))}
    return out


def grouped(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        m = regression_metrics(g["force"].to_numpy(), g["pred"].to_numpy())
        m[by] = key
        rows.append(m)
    return pd.DataFrame(rows).set_index(by).sort_index()
