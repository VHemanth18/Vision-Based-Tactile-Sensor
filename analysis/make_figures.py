"""Generate every data figure of the thesis (vector PDF) from the dataset and the experiment outputs.

    TACTO_DATA_ROOT=/path/to/TACTO_DATASETS python analysis/make_figures.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repository root
sys.path.insert(0, str(ROOT))
from tactile_force.profiles import profile_shape  # noqa: E402

DATA = Path(os.environ.get("TACTO_DATA_ROOT", "/path/to/TACTO_DATASETS"))
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EXP = ROOT / "outputs" / "experiments"
ARC = ROOT / "outputs" / "archive"
E21 = EXP / "exp021_cnn_cal5v5_hex_ref30_last"
E23 = EXP / "exp023_vit_cal5v5_hex_ref30_last"

SHAPES = ["asterisk_indenter", "ball_array_indenter", "hash_indenter", "heart_indenter", "hexagonal_indenter",
          "ring_indenter", "star_indenter", "triangle_indenter"]
NAMES = {s: s.replace("_indenter", "").replace("_", " ").title().replace("Hash", "Hash (#)") for s in SHAPES}
# validated categorical palette (fixed order, one hue per indenter) + markers as secondary encoding
COL = dict(zip(SHAPES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
MRK = dict(zip(SHAPES, ["o", "s", "^", "D", "v", "P", "X", "*"]))
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "savefig.bbox": "tight", "savefig.dpi": 300,
    "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def read_rgb(p):
    import cv2
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def run_dir(shape, f, dur, prof):
    return DATA / shape / f"force_{f}N" / f"duration_{dur}s" / prof


# --------------------------------------------------------------------------- dataset figures
def fig_force_depth():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for s in SHAPES:
        d = pd.read_csv(DATA / s / "force_depth_curve.csv")
        d = d[d.force_N <= 6]
        ax[0].plot(d.max_penetration_mm, d.force_N, color=COL[s], marker=MRK[s], ms=3.5, markevery=4,
                   label=NAMES[s])
        ax[1].plot(d.max_penetration_mm, d.contact_area_mm2, color=COL[s], marker=MRK[s], ms=3.5, markevery=4)
    ax[0].set(xlabel="Gel penetration (mm)", ylabel="Normal force (N)", title="(a) Force–depth calibration")
    ax[1].set(xlabel="Gel penetration (mm)", ylabel="Contact area (mm²)", title="(b) Contact area")
    for a in ax:
        a.set_xlim(left=0)
        a.set_ylim(bottom=0)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, "force_depth")


def fig_profiles():
    u = np.linspace(0, 1, 400)
    kinds = ["square", "trapezoidal", "triangular", "sinusoidal", "sigmoid"]
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.6), sharey=True)
    for a, k in zip(axs, kinds):
        a.plot(u, [profile_shape(k, x) for x in u], color=BLUE)
        a.set_title(k.capitalize())
        a.set_xlabel("Normalised time")
        a.set_xticks([0, 0.5, 1])
    axs[0].set_ylabel("$F/F_{peak}$")
    save(fig, "profiles")


def fig_example_run():
    d = pd.read_csv(run_dir("hexagonal_indenter", 4, 20, "trapezoidal") / "force_depth_calibration.csv")
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 3.6), sharex=True)
    ax[0].plot(d.time_s, d.target_force_N, color=GREY, ls="--", label="Commanded")
    ax[0].plot(d.time_s, d.measured_force_N, color=BLUE, label="Measured (simulator)")
    ax[0].set_ylabel("Force (N)")
    ax[0].legend(loc="upper right")
    ax[0].set_title("Hexagonal indenter, 4 N trapezoidal profile, 20 s")
    ax[1].plot(d.time_s, d.max_penetration_mm.clip(lower=0), color=ORANGE)
    ax[1].set(xlabel="Time (s)", ylabel="Penetration (mm)")
    for a in ax:
        for ph, c in [("idle", "#f3f2ee")]:
            pass
    save(fig, "example_run")


def fig_frames():
    forces = [1, 3, 5]
    fig, axs = plt.subplots(len(forces) * 2, len(SHAPES), figsize=(7.2, 6.2))
    for j, s in enumerate(SHAPES):
        for i, f in enumerate(forces):
            rd = run_dir(s, f, 20, "square")
            fr = sorted((rd / "frames").glob("frame_*.png"))
            ref = np.mean([read_rgb(p) for p in fr[:30]], axis=0)
            img = read_rgb(fr[300])
            axs[2 * i, j].imshow(img)
            diff = np.abs(img - ref).mean(2)
            axs[2 * i + 1, j].imshow(diff, cmap="Blues", vmin=0, vmax=0.25)
            if j == 0:
                axs[2 * i, j].set_ylabel(f"{f} N\nframe", fontsize=7.5)
                axs[2 * i + 1, j].set_ylabel(f"{f} N\n|diff|", fontsize=7.5)
            if i == 0:
                axs[0, j].set_title(NAMES[s], fontsize=7.5)
    for a in axs.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, "frames_grid")


def fig_signal_vs_force():
    rows = []
    for s in SHAPES:
        for f in [1, 2, 3, 4, 5]:
            rd = run_dir(s, f, 20, "square")
            fr = sorted((rd / "frames").glob("frame_*.png"))
            ref = np.mean([read_rgb(p) for p in fr[:30]], axis=0)
            hold = [np.abs(read_rgb(p) - ref).mean() for p in fr[200:440:12]]
            idle = [np.abs(read_rgb(p) - ref).mean() for p in fr[40:100:6]]
            rows.append({"shape": s, "F": f, "sig": np.mean(hold) - np.mean(idle), "sd": np.std(hold)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "signal_vs_force.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for s in SHAPES:
        d = df[df["shape"] == s]
        ax.plot(d.F, d.sig * 255, color=COL[s], marker=MRK[s], ms=5, label=NAMES[s])
    ax.set(xlabel="Peak force (N)", ylabel="Mean |frame − reference| above\nidle level (8-bit units)",
           xticks=[1, 2, 3, 4, 5])
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, loc="lower right")
    save(fig, "signal_vs_force")


def fig_noise():
    rd = run_dir("hexagonal_indenter", 3, 20, "square")
    fr = sorted((rd / "frames").glob("frame_*.png"))
    raw = np.stack([read_rgb(p) for p in fr[:60]]) * 255
    d = (raw[1] - raw[0]).ravel()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].hist(d, bins=np.arange(-40.5, 41.5, 1), color=BLUE, edgecolor="#fcfcfb", linewidth=0.3)
    ax[0].set(xlabel="Pixel difference between two idle frames (8-bit)", ylabel="Count",
              title=f"(a) Frame-to-frame noise, σ = {d.std():.1f}")
    ks = [1, 2, 3, 5, 10, 15, 20, 30, 40, 59]
    target = raw[59]
    err = [np.abs(target - raw[:k].mean(0)).mean() for k in ks]
    ax[1].plot(ks, err, color=ORANGE, marker="o", ms=4)
    ax[1].axhline(np.abs(raw[59] - raw[:59].mean(0)).mean(), color=GREY, ls=":", lw=1)
    ax[1].set(xlabel="Frames averaged into the reference (K)", ylabel="Mean |frame − reference|",
              title="(b) Reference noise vs. K", xscale="log")
    ax[1].set_xticks(ks)
    ax[1].set_xticklabels(ks)
    save(fig, "noise")


# --------------------------------------------------------------------------- result figures
def hist(path):
    return pd.read_csv(path)


def fig_training_curves():
    h21 = hist(E21 / "history.csv")
    h23 = hist(E23 / "history.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax[0].plot(h21.epoch, h21.train_loss, color=BLUE, label="CNN (exp021)")
    ax[0].plot(h23.epoch, h23.train_loss, color=ORANGE, ls="--", label="ViT (exp023)")
    ax[0].set(yscale="log", xlabel="Epoch", ylabel="Training loss (Huber)", title="(a) Training loss")
    ax[0].legend()
    ax[1].plot(h21.epoch, h21.val_mae_N, color=BLUE, marker="o", ms=2.5)
    ax[1].plot(h23.epoch, h23.val_mae_N, color=ORANGE, ls="--", marker="s", ms=2.5)
    ax[1].set(yscale="log", xlabel="Epoch", ylabel="Validation MAE (N)", title="(b) Calibration-val MAE")
    ax[2].plot(h21.epoch, h21.lr, color=BLUE)
    ax[2].plot(h23.epoch, h23.lr, color=ORANGE, ls="--")
    ax[2].set(xlabel="Epoch", ylabel="Learning rate", title="(c) One-cycle schedule")
    fig.tight_layout()
    save(fig, "training_curves")


def fig_parity():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for a, (e, name) in zip(ax, [(E21, "CNN (exp021)"), (E23, "ViT (exp023)")]):
        p = pd.read_csv(e / "eval_test" / "predictions.csv", usecols=["force", "pred"])
        hb = a.hexbin(p.force, p.pred, gridsize=60, bins="log", cmap="Blues", mincnt=1, extent=(0, 5.6, 0, 5.6))
        a.plot([0, 5.6], [0, 5.6], color=INK2, lw=0.8, ls="--")
        a.set(xlabel="True force (N)", title=name, xlim=(0, 5.6), ylim=(0, 5.6), aspect="equal")
        a.grid(False)
    ax[0].set_ylabel("Predicted force (N)")
    fig.colorbar(hb, ax=ax, shrink=0.8, label="Frames (log)")
    save(fig, "parity")


def fig_error_by_group():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k, (col, title) in enumerate([("peak_force", "(a) By peak force"), ("profile", "(b) By force profile")]):
        a = pd.read_csv(E21 / "eval_test" / f"metrics_by_{col}.csv")
        b = pd.read_csv(E23 / "eval_test" / f"metrics_by_{col}.csv")
        x = np.arange(len(a))
        w = 0.38
        ax[k].bar(x - w / 2 - 0.01, a.mae_N, w, color=BLUE, label="CNN (exp021)")
        ax[k].bar(x + w / 2 + 0.01, b.mae_N, w, color=ORANGE, label="ViT (exp023)", hatch="///",
                  edgecolor="#fcfcfb", linewidth=0)
        lab = [f"{v:g} N" for v in a[col]] if col == "peak_force" else [{"trapezoidal": "trapez.", "triangular": "triang.", "sinusoidal": "sinus."}.get(str(v), str(v)) for v in a[col]]
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(lab)
        ax[k].set(ylabel="MAE, all frames (N)", title=title)
        ax[k].grid(axis="x", visible=False)
    ax[0].legend()
    save(fig, "error_by_group")


def smooth(df, kind, w):
    sys.path.insert(0, str(ROOT / "experiments"))
    from smooth_study import smooth as sm
    return sm(df, kind, w)


def fig_timeseries():
    p = pd.read_parquet(E21 / "preds_test_tta.parquet")
    runs = ["hexagonal_indenter|trapezoidal|4N|20s", "hexagonal_indenter|sinusoidal|2N|15s"]
    runs = [r for r in runs if r in set(p.run_id)] or list(p.run_id.unique()[:2])
    fig, ax = plt.subplots(len(runs), 1, figsize=(6.6, 2.2 * len(runs)), sharex=False)
    for a, r in zip(np.atleast_1d(ax), runs):
        d = p[p.run_id == r].sort_values("frame_idx").reset_index(drop=True)
        d["causal"] = smooth(d, "causal_mean", 3)
        d["center"] = smooth(d, "centered_median", 25)
        a.plot(d.time_s, d.force, color=INK, lw=1.2, label="Ground truth")
        a.plot(d.time_s, d.pred, color="#86b6ef", lw=0.8, label="CNN, single frame")
        a.plot(d.time_s, d.causal, color=BLUE, lw=1.0, label="Causal mean (3 frames)")
        a.plot(d.time_s, d.center, color=ORANGE, lw=1.2, ls="--", label="Centred median (25 frames)")
        a.set(ylabel="Force (N)", title=r.replace("hexagonal_indenter|", "Hexagonal, ").replace("|", ", "))
    np.atleast_1d(ax)[-1].set_xlabel("Time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(*np.atleast_1d(ax)[0].get_legend_handles_labels(), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, "timeseries")


def fig_smoothing_window():
    g = pd.read_csv(E21 / "tta_smoothing_eval" / "grid.csv")
    g = g[g.tta]
    base = g[g.smooth == "none"].iloc[0]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for a, (part, title) in zip(ax, [("val", "(a) Calibration-validation runs"), ("test", "(b) Test runs")]):
        for kind, c, m, lab in [("causal_mean", BLUE, "o", "Causal mean (real time)"),
                                ("centered_median", ORANGE, "s", "Centred median (offline)")]:
            d = g[g.smooth == kind].sort_values("w")
            a.plot([1] + list(d.w), [base[f"{part}_contact_mae"]] + list(d[f"{part}_contact_mae"]), color=c,
                   marker=m, ms=4, label=lab)
        a.set(xscale="log", xlabel="Window length (frames)", title=title)
        a.set_xticks([1, 3, 5, 9, 17, 25, 49])
        a.set_xticklabels([1, 3, 5, 9, 17, 25, 49])
    ax[0].set_ylabel("In-contact MAE (N)")
    ax[0].legend()
    save(fig, "smoothing_window")


def fig_progression():
    items = [("Constant (train median)", 1.237), ("exp000 CNN strict", 2.014), ("exp000 ViT strict", 2.721),
             ("exp005 calib, hash val", 0.882), ("exp006 1 run @3 N", 0.519), ("exp011 ViT calib", 0.300),
             ("exp007 CNN calib", 0.096), ("exp021 + ref. avg.", 0.078), ("exp021 + causal", 0.059),
             ("exp021 + centred", 0.043)]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(items))[::-1]
    colors = [GREY] + [ORANGE] * 2 + ["#86b6ef"] * 3 + [BLUE] * 4
    ax.barh(y, [v for _, v in items], color=colors, height=0.62)
    for yi, (_, v) in zip(y, items):
        ax.text(v * 1.08, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in items])
    ax.set(xscale="log", xlabel="In-contact MAE on hexagonal test runs (N, log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.03, 5)
    save(fig, "progression")


def fig_strict_failure():
    p = pd.read_csv(ARC / "experiments" / "exp000_baseline_cnn_vit" / "cnn" / "eval_test" / "predictions.csv",
                    usecols=["run_id", "frame_idx", "time_s", "force", "pred"])
    r = "hexagonal_indenter|trapezoidal|4N|20s"
    r = r if r in set(p.run_id) else p.run_id.iloc[0]
    d = p[p.run_id == r].sort_values("frame_idx")
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    ax.plot(d.time_s, d.force, color=INK, lw=1.2, label="Ground truth")
    ax.plot(d.time_s, d.pred, color=ORANGE, lw=1.0, label="CNN trained on other shapes (exp000)")
    ax.set(xlabel="Time (s)", ylabel="Force (N)", title="Strict unseen-shape setting, hexagonal 4 N")
    ax.legend(loc="upper right")
    save(fig, "strict_failure")


def copy_images():
    src = ROOT / "cad" / "renders"
    dst = OUT / "indenters"
    dst.mkdir(exist_ok=True)
    for p in src.glob("*"):
        shutil.copy2(p, dst / p.name.replace("#_Symbol", "Hash_Symbol").replace(" ", "_"))


def summary_numbers():
    out = {}
    for name, e in [("cnn21", E21), ("vit23", E23)]:
        out[name] = json.loads((e / "eval_test" / "metrics.json").read_text())
    (OUT / "summary_numbers.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- additional analysis figures
def fig_force_distribution():
    idx = pd.read_csv(EXP / "_index" / "index.csv", usecols=["profile", "force", "peak_force"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    bins = np.linspace(0, 5.2, 53)
    kinds = ["square", "trapezoidal", "triangular", "sinusoidal", "sigmoid"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    for k, c in zip(kinds, cols):
        f = idx.loc[(idx.profile == k) & (idx.force > 0.05), "force"]
        ax[0].hist(f, bins=bins, histtype="step", color=c, lw=1.4, label=k)
    ax[0].set(xlabel="Force (N)", ylabel="Frames", title="(a) In-contact force by profile")
    ax[0].legend()
    frac = (idx.force > 0.05).groupby(idx.profile).mean().reindex(kinds)
    ax[1].bar(range(5), frac.values * 100, color=BLUE, width=0.6)
    ax[1].set_xticks(range(5))
    ax[1].set_xticklabels([k[:5] for k in kinds])
    ax[1].set(ylabel="Frames in contact (%)", title="(b) Contact fraction", ylim=(0, 100))
    ax[1].grid(axis="x", visible=False)
    for i, v in enumerate(frac.values * 100):
        ax[1].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=7.5, color=INK2)
    stats = {"n_frames": int(len(idx)), "contact_frac": float((idx.force > 0.05).mean()),
             "contact_frac_by_profile": {k: float(v) for k, v in frac.items()}}
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    save(fig, "force_distribution")


def _phase_table(d):
    d = d.sort_values(["run_id", "frame_idx"]).copy()
    d["dF"] = d.groupby("run_id").force.diff().fillna(0)
    d["phase"] = np.select([d.force <= 0.05, d.dF > 1e-3, d.dF < -1e-3], ["no contact", "loading", "unloading"], "hold")
    return d


def fig_error_analysis():
    p = pd.read_parquet(E21 / "preds_test_plain.parquet")
    p = _phase_table(p)
    p["err"] = p.pred - p.force
    c = p[p.force > 0.05]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].hist(c.err, bins=np.linspace(-0.6, 0.6, 49), color=BLUE, edgecolor="#fcfcfb", linewidth=0.3)
    ax[0].set(xlabel="Prediction error (N)", ylabel="Frames", title="(a) Error distribution, in contact")
    run = c.assign(ae=c.err.abs()).groupby(["run_id", "peak_force"]).ae.mean().reset_index()
    data = [run.loc[run.peak_force == f, "ae"].values for f in [1, 2, 3, 4, 5]]
    bp = ax[1].boxplot(data, positions=range(1, 6), widths=0.5, patch_artist=True, showfliers=True,
                       flierprops=dict(marker="o", ms=3, mfc=GREY, mec=GREY))
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=BLUE)
    for m in bp["medians"]:
        m.set(color=ORANGE, lw=1.6)
    ax[1].set(xlabel="Peak force (N)", ylabel="Per-run in-contact MAE (N)", title="(b) Per-run error (115 runs)")
    save(fig, "error_analysis")
    bins = [0.05, 0.25, 0.5, 1, 2, 3, 4, 5.2]
    c = c.assign(fbin=pd.cut(c.force, bins), ae=c.err.abs())
    t1 = c.groupby("fbin", observed=True).agg(n=("ae", "size"), mae=("ae", "mean"), bias=("err", "mean"))
    t2 = c.groupby("phase").agg(n=("ae", "size"), mae=("ae", "mean"), bias=("err", "mean"))
    t1.to_csv(OUT / "err_by_forcebin.csv")
    t2.to_csv(OUT / "err_by_phase.csv")
    run.describe().to_csv(OUT / "err_per_run.csv")
    print(t1.round(4), t2.round(4), run.ae.describe().round(4), sep="\n")


def fig_gallery():
    times = [0.15, 0.25, 0.32, 0.5, 0.7, 0.78]
    for part, shapes in [(1, SHAPES[:4]), (2, SHAPES[4:])]:
        fig, axs = plt.subplots(len(shapes) * 2, len(times), figsize=(7.0, 8.6))
        for i, s in enumerate(shapes):
            rd = run_dir(s, 5, 20, "trapezoidal")
            fr = sorted((rd / "frames").glob("frame_*.png"))
            lab = pd.read_csv(rd / "force_depth_calibration.csv")
            ref = np.mean([read_rgb(p) for p in fr[:30]], axis=0)
            for j, tt in enumerate(times):
                k = int(tt * (len(fr) - 1))
                img = read_rgb(fr[k])
                axs[2 * i, j].imshow(img)
                axs[2 * i + 1, j].imshow(np.abs(img - ref).mean(2), cmap="Blues", vmin=0, vmax=0.25)
                axs[2 * i, j].set_title(f"{lab.time_s[k]:.1f} s\n{lab.measured_force_N[k]:.2f} N", fontsize=6.5)
            axs[2 * i, 0].set_ylabel(NAMES[s] + "\nframe", fontsize=7.5)
            axs[2 * i + 1, 0].set_ylabel("|diff|", fontsize=7.5)
        for a in axs.ravel():
            a.set_xticks([])
            a.set_yticks([])
            a.grid(False)
        fig.subplots_adjust(wspace=0.04, hspace=0.42)
        save(fig, f"gallery_{part}")


def fig_vit_val_compare():
    rows = []
    for name, path in [("CNN exp007", ARC / "experiments" / "exp007_cnn_cal5v5_os10_hex"),
                       ("CNN exp021", E21), ("ViT exp011", ARC / "experiments" / "exp011_vit_cal5v5_os10_hex"),
                       ("ViT exp023", E23)]:
        h = pd.read_csv(path / "history.csv")
        rows.append((name, h))
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    sty = {"CNN exp007": (BLUE, ":"), "CNN exp021": (BLUE, "-"), "ViT exp011": (ORANGE, ":"), "ViT exp023": (ORANGE, "-")}
    for name, h in rows:
        c, ls = sty[name]
        ax.plot(h.epoch, h.val_mae_N, color=c, ls=ls, label=name)
    ax.set(yscale="log", xlabel="Epoch", ylabel="Validation MAE (N)",
           title="Calibration-validation MAE")
    ax.legend(ncol=2)
    save(fig, "val_compare")



def _cm_panel(ax, cm, labels_x, labels_y, fmt="pct", title=None):
    """Confusion matrix as a single-hue sequential heat map (row-normalised), with values written in the cells."""
    m = cm.to_numpy(dtype=float)
    rown = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] == 0:
                continue
            s = (f"{100 * rown[i, j]:.1f}" if 100 * rown[i, j] >= 0.05 else "<0.1") if fmt == "pct" else f"{int(m[i, j])}"
            ax.text(j, i, s, ha="center", va="center", fontsize=6.5,
                    color="white" if rown[i, j] > 0.55 else "#222222")
    ax.set_xticks(range(len(labels_x)), labels_x, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_y)), labels_y)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.spines[:].set_visible(False); ax.grid(False)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, loc="left")


def fig_force_level_cm():
    d = EXP / "exp021_cnn_cal5v5_hex_ref30_last" / "force_level_classification"
    f = pd.read_csv(d / "frame_confusion_single_frame.csv", index_col=0)
    r = pd.read_csv(d / "run_confusion.csv", index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.45, 1]})
    _cm_panel(ax[0], f, list(f.columns), list(f.index), "pct", "(a) Frame level, single frame (% of row)")
    lv = [f"{int(float(c))} N" for c in r.columns]
    _cm_panel(ax[1], r, lv, lv, "count", "(b) Run level, peak force (runs)")
    fig.tight_layout()
    save(fig, "force_level_cm")


def fig_shape_cm():
    d = sorted(EXP.glob("exp*_cnn_shape_classification"))[-1]
    cm = pd.read_csv(d / "confusion_frames.csv", index_col=0)
    lab = [NAMES.get(s, s) for s in cm.index]
    accf = pd.read_csv(d / "acc_by_force.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    _cm_panel(ax[0], cm, lab, lab, "pct", "(a) Confusion matrix (% of row)")
    names = ["0.05–0.25", "0.25–0.5", "0.5–1", "1–2", "2–3", "3–5"]
    acc = accf["acc"].to_numpy() * 100
    ax[1].bar(range(len(acc)), acc, color="#2a78d6", width=0.6)
    for i, a in enumerate(acc):
        ax[1].text(i, a + 0.15, f"{a:.1f}", ha="center", va="bottom", fontsize=7)
    ax[1].set_ylim(0, 108)
    ax[1].set_xticks(range(len(acc)), names[:len(acc)], rotation=30, ha="right")
    ax[1].set_xlabel("True force (N)"); ax[1].set_ylabel("Frame accuracy (%)")
    ax[1].set_title("(b) Accuracy by force", loc="left")
    fig.tight_layout()
    save(fig, "shape_cm")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for n in sys.argv[1:]:
            globals()[n]()
        sys.exit()
    for f in [copy_images, fig_force_depth, fig_profiles, fig_example_run, fig_noise, fig_signal_vs_force,
              fig_frames, fig_training_curves, fig_parity, fig_error_by_group, fig_timeseries,
              fig_smoothing_window, fig_progression, fig_strict_failure, summary_numbers, fig_force_distribution, fig_error_analysis,
              fig_gallery, fig_vit_val_compare, fig_force_level_cm, fig_shape_cm]:
        f()


