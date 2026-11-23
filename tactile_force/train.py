"""CNN training loop with per-epoch run-balanced subsampling and early stopping."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import FrameForceDataset, subsample_per_run
from .models import build_model
from .plots import training_curves
from .preprocess import in_channels

log = logging.getLogger(__name__)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(frames: pd.DataFrame, cfg, train: bool, seed: int = 0, batch_size: int | None = None) -> DataLoader:
    t = cfg.train
    ds = FrameForceDataset(frames, cfg.preprocess, augment=train and t.augment, seed=seed)
    nw = int(t.num_workers)
    return DataLoader(ds, batch_size=batch_size or t.batch_size, shuffle=train, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), drop_last=train and len(ds) > t.batch_size,
                      persistent_workers=False)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, scale: float) -> np.ndarray:
    model.eval()
    preds = []
    for x, _ in loader:
        preds.append(model(x.to(device, non_blocking=True)).float().cpu().numpy() * scale)
    return np.clip(np.concatenate(preds), 0.0, None) if preds else np.zeros(0)


def train_model(splits: dict[str, pd.DataFrame], cfg, work_dir: Path) -> Path:
    t = cfg.train
    device = pick_device(t.device)
    # train.seed (optional) changes init / subsampling / augmentation only, keeping the split fixed
    tseed = int(t.seed) if t.get("seed") is not None else int(cfg.split.seed)
    torch.manual_seed(tseed)
    model = build_model(cfg, in_channels(cfg.preprocess)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model %s, %.2fM params, device %s", cfg.model.arch, n_params / 1e6, device)

    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=t.lr, epochs=t.epochs, pct_start=0.15,
        steps_per_epoch=max(1, _steps_per_epoch(splits["train"], cfg)))
    loss_fn = nn.MSELoss() if t.get("loss", "huber") == "mse" else nn.HuberLoss(delta=t.huber_delta)
    use_amp = bool(t.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scale = float(t.force_scale)

    # fixed validation subset (same frames every epoch -> comparable numbers)
    val_frames = subsample_per_run(splits["val"], t.max_frames_per_run, seed=123)
    val_loader = make_loader(val_frames, cfg, train=False, batch_size=cfg.evaluate.batch_size)
    y_val = val_frames["force"].to_numpy()

    ckpt = work_dir / "best_model.pt"
    history, best, bad_epochs = [], np.inf, 0
    for epoch in range(1, t.epochs + 1):
        tr_frames = _epoch_frames(splits["train"], cfg, seed=tseed + epoch)
        aug_seed = epoch if t.get("seed") is None else epoch + 1000 * tseed
        loader = make_loader(tr_frames, cfg, train=True, seed=aug_seed)
        model.train()
        t0, run_loss, n_seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) / scale
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
            if step % t.log_every == 0:
                log.info("  epoch %d step %d/%d loss %.5f", epoch, step, len(loader), run_loss / n_seen)

        p_val = predict_loader(model, val_loader, device, scale)
        val_loss = float(loss_fn(torch.tensor(p_val / scale), torch.tensor(y_val / scale)))
        val_mae = float(np.mean(np.abs(p_val - y_val)))
        history.append({"epoch": epoch, "train_loss": run_loss / max(n_seen, 1), "val_loss": val_loss,
                        "val_mae_N": val_mae, "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})
        log.info("epoch %d  train_loss %.5f  val_loss %.5f  val_MAE %.4f N  (%.0fs)", epoch,
                 history[-1]["train_loss"], val_loss, val_mae, history[-1]["sec"])

        state = {"model": model.state_dict(), "cfg": dict(cfg), "epoch": epoch, "val_mae_N": val_mae}
        torch.save(state, work_dir / "last_model.pt")
        if t.get("select", "best") == "last":  # keep the final (fully annealed) model; val only reported
            torch.save(state, ckpt)
            best = min(best, val_mae)
        elif val_mae < best:
            best, bad_epochs = val_mae, 0
            torch.save(state, ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= t.early_stop_patience:
                log.info("Early stopping at epoch %d (best val MAE %.4f N)", epoch, best)
                break

    hist = pd.DataFrame(history)
    hist.to_csv(work_dir / "history.csv", index=False)
    training_curves(hist, work_dir / "training_curves.png")
    log.info("Best checkpoint: %s (val MAE %.4f N)", ckpt, best)
    return ckpt


def _epoch_frames(train_frames: pd.DataFrame, cfg, seed: int) -> pd.DataFrame:
    """Per-epoch run-balanced subsample; runs of train.oversample_shapes are repeated oversample_factor times
    (e.g. the few calibration runs of a new indenter, which would otherwise be <1 % of the frames)."""
    t = cfg.train
    fr = subsample_per_run(train_frames, t.max_frames_per_run, seed=seed)
    shapes, factor = t.get("oversample_shapes") or [], int(t.get("oversample_factor") or 1)
    if shapes and factor > 1:
        extra = fr[fr["shape"].isin(shapes)]
        fr = pd.concat([fr] + [extra] * (factor - 1), ignore_index=True)
    return fr


def _steps_per_epoch(train_frames: pd.DataFrame, cfg) -> int:
    n = _epoch_frames(train_frames, cfg, seed=0).shape[0]
    bs = cfg.train.batch_size
    return n // bs if n > bs else 1


def load_model(ckpt_path: Path, cfg, device: torch.device) -> nn.Module:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg, in_channels(cfg.preprocess))
    model.load_state_dict(state["model"])
    return model.to(device).eval()
