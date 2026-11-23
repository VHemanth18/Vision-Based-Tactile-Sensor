"""Indenter-shape classification (8 classes) from single in-contact tactile frames, with the same CNN backbone
and input recipe as the best force model (exp021: raw + difference to a 30-frame mean reference).

Run-level split, stratified per shape (seed 0): 70 % train / 15 % val / 15 % test runs. Only in-contact frames
(force > contact_threshold) are used. The final (fully annealed) model is evaluated once on the test runs.
Writes a new numbered experiment folder (never overwrites).

    python experiments/shape_classify.py --epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from tactile_force.config import load_config  # noqa: E402
from tactile_force.data import FrameForceDataset, subsample_per_run  # noqa: E402
from tactile_force.models import build_model  # noqa: E402
from tactile_force.preprocess import in_channels  # noqa: E402
from run_exp import next_exp_dir  # noqa: E402

BASE = ROOT / "outputs" / "experiments" / "exp021_cnn_cal5v5_hex_ref30_last" / "config_used.yaml"
INDEX = ROOT / "outputs" / "experiments" / "_index" / "index.csv"
log = logging.getLogger("shape_cls")


class ShapeDS(Dataset):
    def __init__(self, frames, pp, classes, augment, seed):
        self.ds = FrameForceDataset(frames, pp, augment=augment, seed=seed)
        self.y = frames["shape"].map({c: i for i, c in enumerate(classes)}).to_numpy()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, _ = self.ds[i]
        return x, torch.tensor(self.y[i])


def split_runs(idx: pd.DataFrame, seed=0):
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for _, g in idx.groupby("shape"):
        runs = np.array(sorted(g["run_id"].unique()))
        rng.shuffle(runs)
        n = len(runs); nv = nt = int(round(0.15 * n))
        out["test"] += runs[:nt].tolist(); out["val"] += runs[nt:nt + nv].tolist(); out["train"] += runs[nt + nv:].tolist()
    return out


@torch.no_grad()
def predict(model, frames, cfg, classes, dev):
    dl = DataLoader(ShapeDS(frames, cfg.preprocess, classes, False, 0), batch_size=256, num_workers=8)
    model.eval(); P = []
    for x, _ in dl:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            P.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
    return np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--frames_per_run", type=int, default=150)
    a = ap.parse_args()
    cfg = load_config(BASE)
    work = next_exp_dir("cnn_shape_classification")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(work / "train.log"), logging.StreamHandler()])
    (work / "command.txt").write_text(" ".join(sys.argv) + "\n")
    thr = float(cfg.evaluate.contact_threshold)
    idx = pd.read_csv(INDEX)
    idx = idx[idx["force"] > thr].reset_index(drop=True)
    classes = sorted(idx["shape"].unique())
    sp = split_runs(idx)
    json.dump(sp, open(work / "split.json", "w"))
    fr = {k: idx[idx["run_id"].isin(v)].reset_index(drop=True) for k, v in sp.items()}
    log.info("classes %s | runs %s | frames %s", classes, {k: len(v) for k, v in sp.items()},
             {k: len(v) for k, v in fr.items()})

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = build_model(cfg, in_channels(cfg.preprocess))
    model.head[-1] = nn.Linear(model.head[-1].in_features, len(classes))
    model = model.to(dev)
    log.info("params %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)
    bs, fpr = int(cfg.train.batch_size), a.frames_per_run
    steps = len(subsample_per_run(fr["train"], fpr, 0)) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(cfg.train.lr), epochs=a.epochs, pct_start=0.15,
                                                steps_per_epoch=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    val_fr = subsample_per_run(fr["val"], 60, seed=123)
    hist = []
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train()
        dl = DataLoader(ShapeDS(subsample_per_run(fr["train"], fpr, ep), cfg.preprocess, classes, True, ep),
                        batch_size=bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
        tl = n = 0
        for i, (x, y) in enumerate(dl):
            if i >= steps:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = ce(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tl += loss.item() * len(y); n += len(y)
        pv = predict(model, val_fr, cfg, classes, dev)
        yv = val_fr["shape"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        acc = float((pv.argmax(1) == yv).mean())
        hist.append({"epoch": ep, "train_loss": tl / n, "val_acc": acc, "sec": time.time() - t0})
        log.info("epoch %d train_loss %.4f val_acc %.4f (%.0fs)", ep, tl / n, acc, time.time() - t0)
        torch.save({"model": model.state_dict(), "classes": classes, "epoch": ep}, work / "last_model.pt")
    pd.DataFrame(hist).to_csv(work / "history.csv", index=False)

    # test: all in-contact frames of the test runs
    te = fr["test"]
    P = predict(model, te, cfg, classes, dev)
    te = te.assign(pred=[classes[i] for i in P.argmax(1)], conf=P.max(1))
    te.drop(columns=["path", "ref_path"]).to_parquet(work / "preds_test.parquet")
    frame_acc = float((te["pred"] == te["shape"]).mean())
    # run-level: average the class probabilities over the in-contact frames of the run
    run_pred = pd.DataFrame(P, columns=classes).assign(run_id=te["run_id"].values).groupby("run_id").mean()
    run_true = te.groupby("run_id")["shape"].first()
    run_acc = float((run_pred.idxmax(axis=1) == run_true.loc[run_pred.index]).mean())
    bins = [thr, 0.25, 0.5, 1, 2, 3, 5.5]
    te["fbin"] = pd.cut(te["force"], bins)
    ok = te["pred"] == te["shape"]
    by_force = te.assign(ok=ok).groupby("fbin", observed=True)["ok"].agg(n="size", acc="mean")
    by_shape = te.assign(ok=ok).groupby("shape")["ok"].mean()
    cm = pd.crosstab(te["shape"], te["pred"]).reindex(index=classes, columns=classes, fill_value=0)
    cm.to_csv(work / "confusion_frames.csv"); by_force.to_csv(work / "acc_by_force.csv"); by_shape.to_csv(work / "acc_by_shape.csv")
    res = {"frame_acc": frame_acc, "run_acc": run_acc, "n_test_frames": len(te), "n_test_runs": int(len(run_true)),
           "macro_f1": None}
    from sklearn.metrics import f1_score
    res["macro_f1"] = float(f1_score(te["shape"], te["pred"], average="macro"))
    json.dump(res, open(work / "test_metrics.json", "w"), indent=1)
    log.info("TEST %s\n%s\n%s\n%s", res, by_force, by_shape, cm)


if __name__ == "__main__":
    main()
