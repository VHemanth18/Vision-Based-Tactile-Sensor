"""Commands: index, inspect, train, evaluate, predict, compare."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import save_config
from .indexing import build_index, frame_number, summarize
from .metrics import full_report, grouped
from .plots import error_by_force, parity_plot, run_timeseries
from .preprocess import crop_and_resize, read_rgb
from .splits import apply_split, load_split, make_splits, save_split

log = logging.getLogger(__name__)


def _work(cfg) -> Path:
    wd = Path(cfg.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def _idx_dir(cfg) -> Path:
    d = Path(cfg.get("index_dir") or cfg.work_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(cfg) -> pd.DataFrame:
    path = _idx_dir(cfg) / "index.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python main.py index` first")
    return pd.read_csv(path)


def _split_name(cfg) -> str:
    s = cfg.split
    if s.mode != "shape":
        return f"split_{s.mode}.json"
    short = lambda names: "+".join(n.replace("_indenter", "") for n in sorted(names))  # noqa: E731
    val = f"_val-{short(s.val_shapes)}" if s.get("val_shapes") else ""
    calib = f"_calib{int(s.calib_runs_per_force)}" if s.get("calib_runs_per_force") else ""
    if calib and s.get("calib_forces"):
        calib += "at" + "+".join(f"{float(f):g}N" for f in s.calib_forces)
    if calib and s.get("calib_val_runs_per_force"):
        calib += f"v{int(s.calib_val_runs_per_force)}"
    seed = f"_seed{s.seed}" if calib and s.seed != 42 else ""
    return f"split_shape_test-{short(s.test_shapes)}{val}{calib}{seed}.json"


def _splits(cfg) -> dict[str, pd.DataFrame]:
    index = _load_index(cfg)
    split_path = _idx_dir(cfg) / _split_name(cfg)
    if split_path.exists():
        split = load_split(split_path)
        if split["mode"] != cfg.split.mode:
            raise ValueError(f"{split_path} was made with mode={split['mode']}; delete it")
        known = set(split.get("train", [])) | set(split.get("val", [])) | set(split.get("test", []))
        if split["mode"] != "random" and known != set(index["run_id"]):
            raise ValueError(f"{split_path} does not match the current index (runs added or removed). "
                             "Delete it to re-split; models trained on the old split are not comparable.")
    else:
        split = make_splits(index, cfg)
        save_split(split, split_path)
    parts = apply_split(index, split, cfg.split.seed)
    for k, v in parts.items():
        log.info("%-5s: %6d frames, %4d runs, shapes=%s", k, len(v), v["run_id"].nunique(),
                 ",".join(sorted(v["shape"].unique())))
    return parts


# ----------------------------------------------------------------------------- index
def cmd_index(cfg) -> None:
    wd = _idx_dir(cfg)
    index, report = build_index(cfg)
    if index.empty:
        raise RuntimeError("No labelled frames found. Check data.root and the folder layout.")
    index.to_csv(wd / "index.csv", index=False)
    report.to_csv(wd / "coverage.csv", index=False)
    save_config(cfg, wd / "config_used.yaml")
    text = summarize(index, report)
    (wd / "coverage.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {wd / 'index.csv'}")


# ----------------------------------------------------------------------------- inspect
def cmd_inspect(cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = _load_index(cfg)
    pp = cfg.preprocess
    # one high-force frame per indexed shape (the frame closest to that shape's 90th-percentile force)
    picks = []
    for shape, g in index.groupby("shape"):
        target = g["force"].quantile(0.9)
        picks.append(g.iloc[(g["force"] - target).abs().argmin()])
    pick = pd.DataFrame(picks).reset_index(drop=True)
    print(f"Indexed shapes ({len(pick)}): {', '.join(pick['shape'])}")

    full0 = read_rgb(pick.iloc[0]["path"])
    h, w = full0.shape[:2]
    print(f"Raw frame size: {w}x{h} (W x H)")
    if w / h > 1.3 and tuple(pp.crop) == (0.0, 1.0, 0.0, 1.0):
        print("WARNING: frame is much wider than tall -> probably [tactile | depth] panels. "
              "Using the depth panel lets the model read simulator ground truth; set preprocess.crop.")

    fig, axes = plt.subplots(len(pick), 3, figsize=(12, 2.6 * len(pick)), dpi=110)
    axes = np.atleast_2d(axes)
    for row, (_, r) in zip(axes, pick.iterrows()):
        full = read_rgb(r["path"])
        img = crop_and_resize(full, pp)
        ref = crop_and_resize(read_rgb(r["ref_path"]), pp)
        row[0].imshow(full)
        row[0].set_title(f"raw  {r['shape']}  F={r['force']:.2f}N", fontsize=8)
        row[1].imshow(img)
        row[1].set_title("model input (masked + cropped)", fontsize=8)
        d = np.abs(img - ref).mean(axis=2)
        row[2].imshow(d, cmap="magma", vmin=0, vmax=max(d.max(), 1e-3))
        row[2].set_title("|frame - reference|", fontsize=8)
        for a in row:
            a.axis("off")
    out = _idx_dir(cfg) / "inspect.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}. Check: no text overlay and no depth panel in the middle column.")


# ----------------------------------------------------------------------------- evaluation helpers
def _write_eval(df: pd.DataFrame, cfg, out_dir: Path, title: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "predictions.csv", index=False)
    rep = full_report(df, cfg.evaluate.contact_threshold)
    (out_dir / "metrics.json").write_text(json.dumps(rep, indent=2))
    for by in ("shape", "profile", "peak_force", "duration"):
        grouped(df, by).to_csv(out_dir / f"metrics_by_{by}.csv")
    parity_plot(df, out_dir / "parity.png", title)
    error_by_force(df, out_dir / "error_vs_force.png")

    run_dir = out_dir / "runs"
    run_dir.mkdir(exist_ok=True)
    runs = df.groupby("run_id").first().reset_index()
    # spread plots across profiles and shapes
    runs = runs.sample(frac=1.0, random_state=0).sort_values(["profile"], kind="stable")
    chosen = runs.groupby("profile").head(max(1, cfg.evaluate.n_run_plots // 5))["run_id"]
    for rid in chosen.head(cfg.evaluate.n_run_plots):
        safe = rid.replace("|", "_")
        run_timeseries(df[df["run_id"] == rid], run_dir / f"{safe}.png")

    o = rep["overall"]
    msg = f"{title}: MAE {o['mae_N']:.4f} N | RMSE {o['rmse_N']:.4f} N | R2 {o['r2']:.4f}"
    if "in_contact" in rep:
        c = rep["in_contact"]
        msg += f" | in-contact MAE {c['mae_N']:.4f} N ({c['mape_pct']:.1f}%)"
    print(msg)
    print(grouped(df, "shape")[["n", "mae_N", "rmse_N", "r2"]].round(4).to_string())
    return rep


# ----------------------------------------------------------------------------- label check
def _contact_signal(paths: list[str], ref_path: str, pp) -> np.ndarray:
    """Label-free measure of gel deformation per frame.

    |frame - reference| is averaged over colour, blurred to suppress pixel noise, and summarised by
    its 99th percentile, so a small contact patch is not drowned out by the noise of the whole frame.
    """
    import cv2

    ref = crop_and_resize(read_rgb(ref_path), pp)
    out = []
    for p in paths:
        mag = np.abs(crop_and_resize(read_rgb(p), pp) - ref).mean(axis=2)
        out.append(np.percentile(cv2.GaussianBlur(mag, (0, 0), 2.0), 99))
    return np.array(out)


def _best_lag(sig: np.ndarray, force: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (in samples) that maximises the correlation; positive = labels come later than the images."""
    a = (sig - sig.mean()) / (sig.std() + 1e-12)
    b = (force - force.mean()) / (force.std() + 1e-12)
    best = (0, -2.0)
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:len(b) + lag])
        if len(x) > 10:
            c = float(np.mean(x * y))
            if c > best[1]:
                best = (lag, c)
    return best


def cmd_check_labels(cfg, runs_per_shape: int = 3, max_frames: int = 200) -> None:
    """Does the force label rise and fall together with the visible contact in the frames?"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = _load_index(cfg)
    pp = cfg.preprocess
    out_rows, plots = [], []
    for shape, g in index.groupby("shape"):
        runs = (g.groupby("run_id").agg(peak=("peak_force", "first"), fmax=("force", "max"),
                                        src=("label_source", "first"))
                .sort_values(["peak", "fmax"], ascending=False).head(runs_per_shape))
        for rid, info in runs.iterrows():
            r = g[g["run_id"] == rid].sort_values("frame_idx")
            r = r.iloc[np.linspace(0, len(r) - 1, min(max_frames, len(r))).astype(int)]
            sig = _contact_signal(r["path"].tolist(), r["ref_path"].iloc[0], pp)
            force = r["force"].to_numpy()
            dt = float(np.median(np.diff(r["time_s"]))) if len(r) > 1 else 0.0
            corr0 = float(np.corrcoef(sig, force)[0, 1]) if force.std() > 0 and sig.std() > 0 else np.nan
            lag, corr_best = _best_lag(sig, force, max_lag=max(1, int(3.0 / dt)) if dt > 0 else 5)
            low = np.sort(sig)[: max(5, len(sig) // 5)]  # quietest 20 % of frames = no-contact baseline
            has_contact = (sig.max() - np.median(low)) > max(8 * low.std(), 0.002)
            if not has_contact:
                verdict = "NO CONTACT IN IMAGES"
            elif corr0 >= 0.9:
                verdict = "OK"
            elif corr_best >= 0.9 and abs(lag * dt) > 0.1:
                verdict = "SHIFTED"
            else:
                verdict = "MISMATCH"
            out_rows.append({"shape": shape, "run_id": rid, "label_source": info["src"],
                             "peak_force_folder": info["peak"], "max_label_N": round(float(force.max()), 3),
                             "corr": round(corr0, 3), "best_lag_s": round(lag * dt, 2),
                             "corr_at_best_lag": round(corr_best, 3), "verdict": verdict})
            plots.append((rid, r["time_s"].to_numpy(), force, sig, verdict))

    rep = pd.DataFrame(out_rows)
    path = _idx_dir(cfg) / "label_check.csv"
    rep.to_csv(path, index=False)
    print(rep[["shape", "label_source", "peak_force_folder", "max_label_N", "corr", "best_lag_s",
               "verdict"]].to_string(index=False))

    n = len(plots)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), dpi=100, squeeze=False)
    for ax, (rid, t, f, sig, verdict) in zip(axes[:, 0], plots):
        ax.plot(t, f, color="tab:blue", lw=1.5, label="force label (N)")
        ax.set_ylabel("N", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(t, sig, color="tab:orange", lw=1.2, label="image contact signal")
        ax2.set_yticks([])
        ax.set_title(f"{rid}   [{verdict}]", fontsize=9, color="green" if verdict == "OK" else "red")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    fig.legend(["force label (N)", "image contact signal"], loc="upper right")
    fig.tight_layout()
    fig.savefig(_idx_dir(cfg) / "label_check.png")
    plt.close(fig)
    bad = rep[rep["verdict"] != "OK"]
    print(f"\nWrote {path} and label_check.png")
    if len(bad):
        print(f"{bad['shape'].nunique()} shape(s) have runs whose labels do not follow the images: "
              f"{', '.join(sorted(bad['shape'].unique()))}. Do not train on them until fixed.")
    else:
        print("All checked runs: labels follow the images.")


# ----------------------------------------------------------------------------- train / evaluate
def cmd_train(cfg) -> None:
    from .train import train_model

    parts = _splits(cfg)
    save_config(cfg, _work(cfg) / "config_used.yaml")
    train_model(parts, cfg, _work(cfg))
    cmd_evaluate(cfg)


def cmd_evaluate(cfg, ckpt: str | None = None) -> None:
    from .train import load_model, make_loader, pick_device, predict_loader

    parts = _splits(cfg)
    device = pick_device(cfg.train.device)
    model = load_model(Path(ckpt) if ckpt else _work(cfg) / "best_model.pt", cfg, device)
    te = parts["test"].iloc[:: max(1, cfg.evaluate.frame_stride)].reset_index(drop=True)
    loader = make_loader(te, cfg, train=False, batch_size=cfg.evaluate.batch_size)
    te["pred"] = predict_loader(model, loader, device, cfg.train.force_scale)
    _write_eval(te, cfg, _work(cfg) / "eval_test", f"{cfg.model.arch.upper()} ({cfg.preprocess.input_mode})")


# ----------------------------------------------------------------------------- compare
def cmd_compare(dirs: list[str], out: str | None) -> None:
    """Collect test metrics of several trained models into one table and chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for d in map(Path, dirs):
        for mpath in [d / "eval_test" / "metrics.json"]:
            if not mpath.exists():
                print(f"skipped {d}: no eval_test/metrics.json (train the model first)")
                continue
            rep = json.loads(mpath.read_text())
            name = d.name
            row = {"model": name, **{f"all_{k}": v for k, v in rep["overall"].items() if k != "n"}}
            row.update({f"contact_{k}": v for k, v in rep.get("in_contact", {}).items() if k != "n"})
            hist = d / "history.csv"
            if hist.exists():
                h = pd.read_csv(hist)
                row["epochs"], row["train_min"] = len(h), round(h["sec"].sum() / 60, 1)
            ckpt = d / "best_model.pt"
            if ckpt.exists():
                import torch

                sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
                row["params_M"] = round(sum(v.numel() for v in sd.values() if v.dtype.is_floating_point) / 1e6, 3)
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No eval_test/metrics.json under the given folders")
    df = pd.DataFrame(rows).drop_duplicates("model").sort_values("contact_mae_N")
    out_dir = Path(out or "outputs/compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison.csv", index=False)
    cols = [c for c in ["model", "contact_mae_N", "contact_rmse_N", "contact_mape_pct", "contact_r2",
                        "all_mae_N", "params_M", "epochs", "train_min"] if c in df.columns]
    print(df[cols].round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 1.5), dpi=130)
    ax.barh(df["model"], df["contact_mae_N"])
    for y, v in enumerate(df["contact_mae_N"]):
        ax.text(v, y, f" {v:.3f}", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("In-contact MAE on test runs (N), lower is better")
    ax.set_title("Model comparison (same split)")
    ax.grid(True, axis="x", alpha=0.4)
    fig.savefig(out_dir / "comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'comparison.csv'} and comparison.png")


# ----------------------------------------------------------------------------- predict
def cmd_predict(cfg, inp: str, reference: str | None, ckpt: str | None, out: str | None) -> None:
    import torch

    from .preprocess import load_pair, to_input
    from .train import load_model, pick_device

    src = Path(inp)
    exts = {e.lower() for e in cfg.data.image_exts}
    files = sorted((f for f in src.iterdir() if f.suffix.lower() in exts), key=frame_number) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"No images in {src}")
    needs_ref = cfg.preprocess.input_mode != "raw"
    if needs_ref and reference is None:
        if not src.is_dir():
            raise ValueError("Single-image prediction with a diff input mode needs --reference <no-contact frame>")
        reference = str(files[0])
        print(f"Using first frame as no-contact reference: {files[0].name}")

    device = pick_device(cfg.train.device)
    model = load_model(Path(ckpt) if ckpt else _work(cfg) / "best_model.pt", cfg, device)
    ref_img = load_pair(reference, None, cfg.preprocess)[0] if needs_ref else None
    preds, bs = [], cfg.evaluate.batch_size
    with torch.no_grad():
        for i in range(0, len(files), bs):
            batch = [to_input(load_pair(str(f), None, cfg.preprocess)[0], ref_img, cfg.preprocess) for f in files[i:i + bs]]
            x = torch.from_numpy(np.stack(batch)).to(device)
            preds.append(np.clip(model(x).float().cpu().numpy() * cfg.train.force_scale, 0, None))
    df = pd.DataFrame({"file": [f.name for f in files], "pred_force_N": np.concatenate(preds)})
    run_dir = src.parent.parent if src.is_file() else src.parent  # .../<shape>/.../<profile>
    tag = "_".join(run_dir.parts[-4:]) + (f"_{src.stem}" if src.is_file() else "")
    out_path = Path(out) if out else _work(cfg) / "predictions" / f"{tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if len(df) == 1:
        print(f"{df.iloc[0]['file']}: {df.iloc[0]['pred_force_N']:.3f} N")
    else:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
        ax.plot([frame_number(f) for f in files], df["pred_force_N"])
        ax.set_xlabel("frame")
        ax.set_ylabel("Predicted force (N)")
        ax.grid(True)
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)
        print(f"{len(df)} frames, max predicted {df['pred_force_N'].max():.3f} N")
    print(f"Wrote {out_path}")
