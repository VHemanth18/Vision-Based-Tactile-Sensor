"""Create a small fake dataset with the same layout and frame format as the real one.

Use it to check the installation and the pipeline before running on the real data:
    python tools/make_synthetic_dataset.py --out synthetic_dataset
    python main.py all --config configs/cnn.yaml data.root=synthetic_dataset index_dir=outputs/synthetic \
        work_dir=outputs/synthetic train.epochs=3 preprocess.mask_top_px=22 "preprocess.crop=[0,0.5,0,1]"

Frames imitate demo_pybullet_digit_force.py: [tactile RGB | depth colormap] + text overlay,
force_depth_calibration.csv logged at 240 Hz, frames rendered every 8th control step (30 fps).
A few runs deliberately lack a CSV (tests the target-profile fallback) or a frames folder.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tactile_force.profiles import target_force  # noqa: E402

H, W = 160, 120
DT, RENDER_EVERY = 1 / 240, 8


def shape_mask(shape: str, radius: float) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    c = (W // 2, H // 2)
    r = max(1, int(radius))
    if shape == "ball_array_indenter":
        for dx in (-1, 1):
            for dy in (-1, 1):
                cv2.circle(m, (c[0] + dx * 22, c[1] + dy * 22), max(1, r // 2), 255, -1)
    elif shape == "cross_indenter":
        cv2.rectangle(m, (c[0] - r, c[1] - r // 4), (c[0] + r, c[1] + r // 4), 255, -1)
        cv2.rectangle(m, (c[0] - r // 4, c[1] - r), (c[0] + r // 4, c[1] + r), 255, -1)
    elif shape == "star_indenter":
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.45
            a = k * np.pi / 5
            pts.append((c[0] + rr * np.sin(a), c[1] - rr * np.cos(a)))
        cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    else:  # hexagonal and default
        pts = [(c[0] + r * np.cos(k * np.pi / 3), c[1] + r * np.sin(k * np.pi / 3)) for k in range(6)]
        cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return m


def render(shape: str, force: float, bg: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    depth = np.zeros((H, W), np.float32)
    color = bg.copy()
    if force > 0.01:
        mask = shape_mask(shape, 8 + 7 * np.sqrt(force)).astype(np.float32) / 255
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0 + 0.3 * force)
        depth = mask * 0.0004 * force
        gy, gx = np.gradient(mask)
        shade = np.stack([gx, gy, -gx], axis=2) * 60 * force + mask[..., None] * np.array([10, 25, 40]) * force / 5
        color = np.clip(color + shade, 0, 255)
    color = np.clip(color + rng.normal(0, 1.5, color.shape), 0, 255).astype(np.uint8)
    return color, depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_dataset")
    ap.add_argument("--shapes", nargs="+", default=["ball_array_indenter", "cross_indenter",
                                                    "hexagonal_indenter", "star_indenter"])
    ap.add_argument("--forces", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--durations", nargs="+", type=int, default=[10, 15])
    ap.add_argument("--frame_stride", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.stack([90 + 40 * xx / W, 110 + 30 * yy / H, 150 - 20 * xx / W], axis=2).astype(np.float32)
    root = Path(args.out)
    profiles = ["square", "trapezoidal", "triangular", "sinusoidal", "sigmoid"]
    n = 0
    for shape in args.shapes:
        for F in args.forces:
            for T in args.durations:
                for prof in profiles:
                    run = root / shape / f"force_{F}N" / f"duration_{T}s" / prof
                    run.mkdir(parents=True, exist_ok=True)
                    if shape == "star_indenter" and F == 5:  # mimic missing frames folders
                        continue
                    t = np.arange(0, T + 1e-9, DT)
                    tgt = target_force(prof, F, T, t)
                    meas = np.convolve(tgt, np.ones(6) / 6, mode="same") + rng.normal(0, 0.01, len(t)) * (tgt > 0)
                    meas = np.clip(meas, 0, None)
                    fdir = run / "frames"
                    fdir.mkdir(exist_ok=True)
                    gel = np.full(len(t), np.nan)
                    for k in range(0, len(t), RENDER_EVERY):
                        vid = k // RENDER_EVERY
                        color, depth = render(shape, meas[k], bg, rng)
                        gel[k] = depth.max() * 1e3
                        if vid % args.frame_stride:
                            continue
                        dvis = cv2.applyColorMap((np.clip(depth / 0.002, 0, 1) * 255).astype(np.uint8),
                                                 cv2.COLORMAP_JET)
                        frame = np.hstack([cv2.cvtColor(color, cv2.COLOR_RGB2BGR), dvis])
                        cv2.putText(frame, f"t={t[k]:5.2f}s ref={tgt[k]:4.2f}N F={meas[k]:4.2f}N", (5, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.imwrite(str(fdir / f"frame_{vid:05d}.png"), frame)
                    if shape == "hexagonal_indenter" and F == 3:  # mimic runs without label CSV
                        continue
                    phase = np.where(tgt <= 1e-6, "idle", "pressing")
                    pd.DataFrame({"time_s": t, "phase": phase, "target_force_N": tgt, "measured_force_N": meas,
                                  "z_height_m": 0.03 - meas / 2500, "gel_depth_mm": gel}
                                 ).to_csv(run / "force_depth_calibration.csv", index=False)
                    n += 1
    print(f"Wrote {n} labelled runs to {root.resolve()}")


if __name__ == "__main__":
    main()
