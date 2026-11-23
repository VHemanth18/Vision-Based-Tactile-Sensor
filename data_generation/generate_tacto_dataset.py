"""Force-controlled TACTO/PyBullet dataset generation (8 indenters x 5 profiles x 5 forces x 5 durations).

Adapted from TACTO examples/demo_pybullet_digit.py (MIT licence, Facebook, Inc.).
For every indenter: find the contact height z0, record the force-depth curve F = phi(delta) on
0-3 mm, then run every profile/force/duration with feed-forward depth control
delta*(t) = phi^-1(F*(t)), z(t) = z0 - delta*(t), saving each DIGIT frame with its label row.

    python generate_tacto_dataset.py experiment.shapes=[hash_indenter] experiment.out_root=/path/to/TACTO_DATASETS
"""
import csv
import json
import logging
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):          # pybulletX uses the removed np.float alias
    np.float = float

import cv2
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pybulletX as px
import tacto
from omegaconf import OmegaConf

from profiles import target_force     # tactile_force/profiles.py (Listing A.2)

log = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent

LOCKED_X, LOCKED_Y = -0.022, 0.000    # indenter centred over the gel; only z is commanded
LOCKED_EULER = [-1.6, 0.0, -0.7]
FPS = 30                              # one rendered frame every 1/30 s
SETTLE_STEPS = 10                     # physics steps after each pose change
HOVER_MM = 0.3                        # idle height above the gel
CURVE_STEP_MM, CURVE_MAX_MM = 0.05, 3.0
DEPTH_EPS_M = 1e-5                    # gel deformation counted as contact
HOLD_SLOPE = 0.02                     # |dF/dt| < 0.02 * peak per second counts as holding


def normal_force(obj_id, digit_id):
    """Sum of the normal contact forces between indenter and gel, projected on z."""
    return sum(c[9] * abs(c[7][2]) for c in p.getContactPoints(bodyA=obj_id, bodyB=digit_id))


def set_height(obj_id, quat, z):
    p.resetBasePositionAndOrientation(obj_id, [LOCKED_X, LOCKED_Y, z], quat)
    for _ in range(SETTLE_STEPS):
        p.stepSimulation()


def contact_stats(depth, pixel_area_mm2):
    """Maximum penetration (mm) and contact area (mm^2) from the TACTO depth image."""
    d = np.asarray(depth[0] if isinstance(depth, list) else depth, dtype=float)
    return float(d.max() * 1000.0), float((d > DEPTH_EPS_M).sum() * pixel_area_mm2)


def find_contact_height(obj_id, digit_id, quat, z_start=0.045, step=1e-5, threshold=1e-3):
    """Lower the indenter until the first contact force appears; returns z0."""
    z = z_start
    while z > 0.010:
        set_height(obj_id, quat, z)
        if normal_force(obj_id, digit_id) >= threshold:
            return z + step
        z -= step
    raise RuntimeError("no contact found above z = 10 mm")


def force_depth_curve(obj_id, digit_id, quat, sensor, z0, pixel_area_mm2, path):
    """Step the indenter 0-3 mm into the gel and record F, contact area and penetration."""
    rows = []
    for d_mm in np.arange(0.0, CURVE_MAX_MM + 1e-9, CURVE_STEP_MM):
        z = z0 - d_mm / 1000.0
        set_height(obj_id, quat, z)
        _, depth = sensor.render()
        pen, area = contact_stats(depth, pixel_area_mm2)
        rows.append([d_mm, normal_force(obj_id, digit_id), area, pen, z])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["indentation_mm", "force_N", "contact_area_mm2", "max_penetration_mm", "z_height_m"])
        w.writerows(rows)
    curve = np.array(rows)
    return curve[:, 0], curve[:, 1]


def phases(target, peak):
    """idle / pressing / holding / releasing from the slope of the commanded force."""
    slope = np.gradient(target, 1.0 / FPS)
    return np.where(target <= 0.0, "idle",
                    np.where(np.abs(slope) < HOLD_SLOPE * peak, "holding",
                             np.where(slope > 0.0, "pressing", "releasing")))


def run_one(obj_id, digit_id, quat, sensor, z0, curve, shape, profile, force, duration, out_dir, pixel_area_mm2):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    depth_mm, force_curve = curve
    t = np.round(np.arange(int(round(duration * FPS)) + 1) / FPS, 6)
    target = target_force(profile, force, duration, t)
    phase = phases(target, force)
    video = None
    rows = []
    for i, ti in enumerate(t):
        if target[i] <= 0.0:
            ind_mm = -HOVER_MM
        else:                                             # feed-forward: invert F = phi(delta)
            ind_mm = float(np.interp(target[i], force_curve, depth_mm))
        z = z0 - ind_mm / 1000.0
        set_height(obj_id, quat, z)
        measured = normal_force(obj_id, digit_id)
        color, depth = sensor.render()
        pen, area = contact_stats(depth, pixel_area_mm2)
        frame = cv2.cvtColor(np.asarray(color[0]).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "frames" / f"frame_{i + 1:05d}.png"), frame)
        if video is None:
            h, w = frame.shape[:2]
            video = cv2.VideoWriter(str(out_dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        video.write(frame)
        rows.append([ti, i + 1, phase[i], round(target[i], 6), round(measured, 6),
                     round(z, 8), round(ind_mm, 6), round(pen, 6), round(area, 4)])
    video.release()

    with open(out_dir / "force_depth_calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "frame", "phase", "target_force_N", "measured_force_N", "z_height_m",
                    "indentation_mm", "max_penetration_mm", "contact_area_mm2"])
        w.writerows(rows)

    meas = np.array([r[4] for r in rows])
    err = meas - target
    hold = meas[[r[2] == "holding" for r in rows]]
    summary = {"shape": shape, "profile": profile, "force_N": force, "duration_s": duration,
               "n_frames": len(rows), "rmse_N": float(np.sqrt(np.mean(err ** 2))),
               "max_abs_err_N": float(np.abs(err).max()),
               "overshoot_pct": float(max(0.0, (meas.max() - force) / force * 100.0)),
               "hold_ripple_N": float(hold.max() - hold.min()) if hold.size else 0.0,
               "peak_measured_N": float(meas.max())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(8, 4))
    plt.plot(t, target, "--", label="target")
    plt.plot(t, meas, label="measured")
    plt.xlabel("Time (s)"), plt.ylabel("Force (N)"), plt.legend(), plt.grid(True)
    plt.title(f"{shape} {profile} {force:g} N {duration:g} s")
    plt.savefig(out_dir / "force_plot.png", dpi=120)
    plt.close()
    return summary


@hydra.main(config_path="conf", config_name="digit", version_base=None)
def main(cfg):
    ex = cfg.experiment
    out_root = Path(ex.out_root)
    px.init(mode=p.DIRECT)
    p.setPhysicsEngineParameter(numSolverIterations=400, fixedTimeStep=1.0 / 240.0)
    bg = cv2.imread(str(SCRIPT_DIR / "conf" / "bg_digit_240_320.jpg"))
    sensor = tacto.Sensor(**cfg.tacto, background=bg)
    digit = px.Body(**cfg.digit)
    sensor.add_camera(digit.id, [-1])
    quat = p.getQuaternionFromEuler(LOCKED_EULER)

    for shape in ex.shapes:
        obj_cfg = OmegaConf.merge(cfg.object, {"urdf_path": f"objects/{shape}.urdf"})
        obj = px.Body(**obj_cfg)
        sensor.add_body(obj)
        z0 = find_contact_height(obj.id, digit.id, quat)
        shape_dir = out_root / shape
        shape_dir.mkdir(parents=True, exist_ok=True)
        curve = force_depth_curve(obj.id, digit.id, quat, sensor, z0, ex.pixel_area_mm2,
                                  shape_dir / "force_depth_curve.csv")
        log.info("%s: z0 = %.5f m, F(3 mm) = %.2f N", shape, z0, curve[1][-1])

        with open(shape_dir / "collection_log.csv", "w", newline="") as f:
            w = None
            for profile in ex.profiles:
                for force in ex.forces_N:
                    for duration in ex.durations_s:
                        run_dir = shape_dir / f"force_{force:g}N" / f"duration_{duration:g}s" / profile
                        s = run_one(obj.id, digit.id, quat, sensor, z0, curve, shape, profile, force,
                                    duration, run_dir, ex.pixel_area_mm2)
                        if w is None:
                            w = csv.DictWriter(f, fieldnames=list(s))
                            w.writeheader()
                        w.writerow(s)
                        log.info("%s %s %g N %g s: rmse %.4f N", shape, profile, force, duration, s["rmse_N"])
        p.removeBody(obj.id)


if __name__ == "__main__":
    main()
