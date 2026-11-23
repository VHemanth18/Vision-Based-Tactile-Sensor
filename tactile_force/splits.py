"""Train/val/test splits grouped by run, so no run contributes frames to two splits."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def make_splits(index: pd.DataFrame, cfg) -> dict[str, list[str]]:
    s = cfg.split
    rng = np.random.default_rng(s.seed)
    runs = index.groupby("run_id").first()[["shape"]].reset_index()

    if s.mode == "random":
        log.warning("split.mode=random mixes frames of one run across splits: metrics will be optimistic.")
        return {"mode": "random"}

    test, val, train = [], [], []
    if s.mode == "shape":
        test_shapes = list(s.test_shapes)
        val_shapes = list(s.get("val_shapes") or [])
        available = sorted(runs["shape"].unique())
        missing = set(test_shapes + val_shapes) - set(available)
        if missing:
            raise ValueError(f"Shapes not in the index: {sorted(missing)}. Available: {available}")
        if set(test_shapes) & set(val_shapes):
            raise ValueError("A shape cannot be in both split.test_shapes and split.val_shapes")
        forces = index.groupby("shape")["peak_force"].unique()
        all_levels = set(np.concatenate(forces.to_numpy()))
        for sh in test_shapes + val_shapes:
            lacking = [f"{x:g}" for x in sorted(all_levels - set(forces[sh]))]
            if lacking:
                log.warning("Held-out shape %s has no runs at %s N; its metrics do not cover those forces",
                            sh, ", ".join(lacking))
        test = runs.loc[runs["shape"].isin(test_shapes), "run_id"].tolist()
        calib: list[str] = []
        calib_val: list[str] = []
        k = int(s.get("calib_runs_per_force") or 0)
        kv = int(s.get("calib_val_runs_per_force") or 0)
        if k:  # few-shot calibration: k labelled runs per (test shape, peak force) move into training
            meta = index.groupby("run_id").first()[["shape", "peak_force"]]
            only = {float(f) for f in (s.get("calib_forces") or [])}  # e.g. [3]: calibrate at 3 N only
            for (_, peak), grp in meta.loc[test].groupby(["shape", "peak_force"]):
                if only and float(peak) not in only:
                    continue
                perm = rng.permutation(grp.index.to_numpy())
                calib += perm[:k].tolist()
                calib_val += perm[k:k + kv].tolist()  # further calibration runs, used only for early stopping
            test = [r for r in test if r not in set(calib) | set(calib_val)]
            log.info("Few-shot calibration: %d runs of the test shape(s) moved to train: %s", len(calib), calib)
            if calib_val:
                log.info("Calibration-validation runs (early stopping only): %s", calib_val)
        if calib_val:  # select the checkpoint on held-out runs of the calibrated shape
            val = calib_val
            train = runs.loc[~runs["shape"].isin(test_shapes + val_shapes), "run_id"].tolist()
        elif val_shapes:  # validation on unseen shapes -> early stopping selects for generalisation
            val = runs.loc[runs["shape"].isin(val_shapes), "run_id"].tolist()
            train = runs.loc[~runs["shape"].isin(test_shapes + val_shapes), "run_id"].tolist()
        else:
            log.warning("split.val_shapes is empty: validation uses runs of the training shapes, so early "
                        "stopping optimises seen-shape accuracy, not unseen-shape accuracy.")
            for _, grp in runs[~runs["shape"].isin(test_shapes)].groupby("shape"):
                ids = rng.permutation(grp["run_id"].to_numpy())
                n_val = max(1, int(round(len(ids) * s.val_frac))) if len(ids) > 2 else 0
                val += ids[:n_val].tolist()
                train += ids[n_val:].tolist()
        train += calib
        log.info("Shape split  train: %s | val: %s | test: %s",
                 sorted(set(available) - set(test_shapes) - set(val_shapes)), val_shapes or "(runs of train shapes)",
                 test_shapes)
    elif s.mode == "run":
        # stratify by shape so every indenter appears in every split
        for _, grp in runs.groupby("shape"):
            ids = rng.permutation(grp["run_id"].to_numpy())
            n = len(ids)
            n_test = int(round(n * s.test_frac)) if n >= 3 else 0
            n_val = int(round(n * s.val_frac)) if n >= 3 else 0
            test += ids[:n_test].tolist()
            val += ids[n_test:n_test + n_val].tolist()
            train += ids[n_test + n_val:].tolist()
    else:
        raise ValueError(f"Unknown split.mode: {s.mode}")

    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test)), "split leakage"
    return {"mode": s.mode, "train": sorted(train), "val": sorted(val), "test": sorted(test)}


def apply_split(index: pd.DataFrame, split: dict, seed: int = 0) -> dict[str, pd.DataFrame]:
    if split["mode"] == "random":
        perm = np.random.default_rng(seed).permutation(len(index))
        n = len(index)
        a, b = int(0.7 * n), int(0.85 * n)
        return {"train": index.iloc[perm[:a]], "val": index.iloc[perm[a:b]], "test": index.iloc[perm[b:]]}
    return {k: index[index["run_id"].isin(split[k])].reset_index(drop=True) for k in ("train", "val", "test")}


def save_split(split: dict, path: Path) -> None:
    path.write_text(json.dumps(split, indent=1))


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())
