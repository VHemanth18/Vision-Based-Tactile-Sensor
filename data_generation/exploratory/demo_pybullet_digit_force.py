import csv
import time
import logging
from pathlib import Path
import numpy as np

if not hasattr(np, "float"):
    np.float = float

import cv2
import hydra
from omegaconf import OmegaConf
import pybullet as p
import pybulletX as px
import tacto
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent

# Hardcoded pose constraints
LOCKED_X = -0.022
LOCKED_Y = 0.000
LOCKED_QUAT = None  # Will be initialized in main()


def get_normal_force(object_id, digit_id):
    contact_points = p.getContactPoints(bodyA=object_id, bodyB=digit_id)
    total_fz = 0.0
    for contact in contact_points:
        normal_force = contact[9]
        normal_vector_z = contact[7][2]
        total_fz += normal_force * abs(normal_vector_z)
    return total_fz


def write_video_frame(video_writer, color_frames, video_path, fps):
    frame = color_frames[0]
    frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if video_writer is None:
        h, w = frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (w, h))
    video_writer.write(frame_bgr)
    return video_writer


def find_contact_surface(obj_id, digit_id, start_z=0.0450, step_size=0.0001, threshold_force=0.02):
    """Lowers the indenter until initial contact is detected, then backs off slightly."""
    log.info("Homing indenter to find initial contact surface...")
    z = start_z
    p.resetBasePositionAndOrientation(obj_id, [LOCKED_X, LOCKED_Y, z], LOCKED_QUAT)
    while True:
        p.stepSimulation()
        f = get_normal_force(obj_id, digit_id)
        if f >= threshold_force:
            log.info(f"Contact detected at Z = {z:.5f}m with {f:.4f}N")
            z += step_size * 2  # Back off slightly to zero out pre-load
            p.resetBasePositionAndOrientation(obj_id, [LOCKED_X, LOCKED_Y, z], LOCKED_QUAT)
            p.stepSimulation()
            break
        if z <= 0.010:
            log.warning("Reached minimum Z safety limit before contact!")
            break
        z -= step_size
        p.resetBasePositionAndOrientation(obj_id, [LOCKED_X, LOCKED_Y, z], LOCKED_QUAT)
    return z

def calibrate_force_depth(obj_id, digit_id, z_start, force_max_N, out_dir, max_depth=0.0025, steps=50):
    """
    Sweeps depth to record raw unmultiplied get_normal_force() vs depth,
    saves the force-depth curve, and derives the real force scale if the mesh under-reports.
    """
    log.info("Running slow force-vs-depth calibration sweep...")
    depths = np.linspace(0.0, max_depth, steps)
    raw_forces = []

    sweep_path = out_dir / "force_depth_calibration.csv"
    with open(sweep_path, mode="w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["depth_m", "z_height_m", "raw_force_N"])
        for d in depths:
            z_test = z_start - d
            p.resetBasePositionAndOrientation(obj_id, [LOCKED_X, LOCKED_Y, z_test], LOCKED_QUAT)
            for _ in range(10):
                p.stepSimulation()
            rf = get_normal_force(obj_id, digit_id)
            raw_forces.append(rf)
            writer.writerow([f"{d:.6f}", f"{z_test:.6f}", f"{rf:.6f}"])

    # Return indenter to z_start and settle
    p.resetBasePositionAndOrientation(obj_id, [LOCKED_X, LOCKED_Y, z_start], LOCKED_QUAT)
    for _ in range(20):
        p.stepSimulation()

    peak_raw = max(raw_forces) if raw_forces else 0.0
    # If the mesh geometry under-reports raw force at full safe depth, scale proportionally
    if 1e-4 < peak_raw < (force_max_N * 1.2):
        force_scale = (force_max_N * 1.2) / peak_raw
    else:
        force_scale = 1.0

    log.info(f"Sweep peak raw force: {peak_raw:.4f}N at {max_depth*1000:.1f}mm -> Derived scale: {force_scale:.3f}x")
    return force_scale


class PIDForceController:
    def __init__(self, kp, ki, kd, integral_limit, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self._has_prev = False

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self._has_prev = False

    def update(self, error, dt):
        self.integral = float(
            np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        )
        derivative = 0.0 if not self._has_prev else (error - self.prev_error) / dt
        self.prev_error = error
        self._has_prev = True
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.output_limit, self.output_limit))


def get_experiment_params(cfg):
    exp = cfg.get("experiment", {}) if hasattr(cfg, "get") else {}
    indenter_name = OmegaConf.select(cfg, "experiment.indenter_name", default="Heart_Shape_Indenter")
    waveform = OmegaConf.select(cfg, "experiment.waveform", default="Square")
    force_max_N = float(OmegaConf.select(cfg, "experiment.force_max_N", default=1.0))
    duration_s = float(OmegaConf.select(cfg, "experiment.duration_s", default=10.0))
    return indenter_name, waveform, force_max_N, duration_s


@hydra.main(config_path="conf", config_name="digit", version_base=None)
def main(cfg):
    global LOCKED_QUAT
    indenter_name, waveform, force_max_N, duration_s = get_experiment_params(cfg)

    # ---------------- OUTPUT DIRECTORY ----------------
    out_dir = (
        SCRIPT_DIR
        / "dataset"
        / f"{indenter_name}_indenter"
        / f"force_{force_max_N:g}N"
        / f"duration_{duration_s:g}s"
        / waveform.lower()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing outputs to: {out_dir}")

    log.info("Initializing PyBullet World")
    px.init()

    PHYSICS_RATE = 240.0
    DT = 1.0 / PHYSICS_RATE
    p.setPhysicsEngineParameter(
        numSolverIterations=400, fixedTimeStep=DT, contactERP=0.2, frictionERP=0.2
    )
    p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    bg_path = SCRIPT_DIR / "conf" / "bg_digit_240_320.jpg"
    bg = cv2.imread(str(bg_path)) if bg_path.exists() else None

    digits = tacto.Sensor(**cfg.tacto, background=bg)
    digit_body = px.Body(**cfg.digit)
    digits.add_camera(digit_body.id, [-1])

    obj = px.Body(**cfg.object)
    digits.add_body(obj)

    # ---------------- DYNAMIC PHYSICS SCALING ----------------
    # Scale stiffness dynamically so 1N remains soft
    CONTACT_STIFFNESS = int(300000 + (force_max_N * 150000))
    CONTACT_DAMPING = int(500 + (force_max_N * 100))

    for body_id in (obj.id, digit_body.id):
        p.changeDynamics(
            body_id, -1, contactStiffness=CONTACT_STIFFNESS, contactDamping=CONTACT_DAMPING, restitution=0.0
        )

    # ---------------- POSE SETUP ----------------
    LOCKED_QUAT = p.getQuaternionFromEuler([-1.6, 0.0, -0.7])

    # Auto-zero the indenter to find exact starting height
    z_start = find_contact_surface(obj.id, digit_body.id, start_z=0.0450)

    # Characterize raw force vs depth for this shape before starting the waveform
    force_scale = calibrate_force_depth(obj.id, digit_body.id, z_start, force_max_N, out_dir)

    TOTAL_TIME_S = duration_s
    TOTAL_STEPS = int(round(TOTAL_TIME_S * PHYSICS_RATE))

    VIDEO_FPS = 30
    VIDEO_PLAYBACK_FPS = 30
    STEPS_PER_FRAME = int(round(PHYSICS_RATE / VIDEO_FPS))

    F_MIN_REF = 0.0
    F_MAX_REF = force_max_N

    # ---------------- DYNAMIC PID SCALING ----------------
    # Normalize gains by force_max_N so tracking response is identical from 1N to 5N
    KP = 8.0e-5 / force_max_N
    KI = 4.0e-5 / force_max_N
    KD = 8.0e-6 / force_max_N
    TRIM_INTEGRAL_LIMIT = 2.0 * force_max_N

    MAX_DELTA_Z = 0.00010

    pid = PIDForceController(
        kp=KP, ki=KI, kd=KD, integral_limit=TRIM_INTEGRAL_LIMIT, output_limit=MAX_DELTA_Z
    )

    FORCE_FILTER_ALPHA = 0.25

    z_command = z_start
    filtered_force = 0.0

    log_path = out_dir / "force_ramp_log.csv"
    csv_file = open(log_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["time_s", "phase", "z_height_m", "raw_force_N", "target_force_N", "measured_force_N", "filtered_force_N"]
    )

    video_path = out_dir / "tacto_force_ramp.mp4"
    video_writer = None

    gui_video_path = out_dir / "gui_capture.mp4"
    gui_log_id = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=str(gui_video_path))

    times, positions, measureds, references = [], [], [], []
    print_interval_steps = max(1, int(PHYSICS_RATE // 10))

    prev_phase = "pre_hold"
    wf_type = waveform.lower()

    # Helper for smooth Sigmoid normalized between 0 and 1
    def get_sigmoid_norm(x):
        s = 1.0 / (1.0 + np.exp(-12.0 * (x - 0.5)))
        s0 = 1.0 / (1.0 + np.exp(6.0))
        s1 = 1.0 / (1.0 + np.exp(-6.0))
        return (s - s0) / (s1 - s0)

    frame_count = 1  # <--- Added explicit sequential frame counter

    for step in range(TOTAL_STEPS + 1):
        elapsed = step * DT
        t_norm = elapsed / duration_s

        if wf_type == "square":
            # 20% wait, 60% hold max, 20% wait
            if t_norm < 0.2:
                phase, ref_force = "pre_hold", F_MIN_REF
            elif t_norm <= 0.8:
                phase, ref_force = "holding", F_MAX_REF
            else:
                phase, ref_force = "post_hold", F_MIN_REF

        elif wf_type == "trapezoidal":
            # 20% wait, 20% ramp up, 20% hold max, 20% ramp down, 20% wait
            if t_norm < 0.2:
                phase, ref_force = "pre_hold", F_MIN_REF
            elif t_norm < 0.4:
                phase = "pressing_in"
                frac = (t_norm - 0.2) / 0.2
                ref_force = F_MIN_REF + (F_MAX_REF - F_MIN_REF) * frac
            elif t_norm < 0.6:
                phase, ref_force = "holding", F_MAX_REF
            elif t_norm < 0.8:
                phase = "coming_out"
                frac = (t_norm - 0.6) / 0.2
                ref_force = F_MAX_REF - (F_MAX_REF - F_MIN_REF) * frac
            else:
                phase, ref_force = "post_hold", F_MIN_REF

        elif wf_type == "triangular":
            # 20% wait, 30% ramp up, 30% ramp down, 20% wait
            if t_norm < 0.2:
                phase, ref_force = "pre_hold", F_MIN_REF
            elif t_norm < 0.5:
                phase = "pressing_in"
                frac = (t_norm - 0.2) / 0.3
                ref_force = F_MIN_REF + (F_MAX_REF - F_MIN_REF) * frac
            elif t_norm < 0.8:
                phase = "coming_out"
                frac = (t_norm - 0.5) / 0.3
                ref_force = F_MAX_REF - (F_MAX_REF - F_MIN_REF) * frac
            else:
                phase, ref_force = "post_hold", F_MIN_REF

        elif wf_type == "sinusoidal":
            # 20% wait, 60% sine wave, 20% wait
            if t_norm < 0.2:
                phase, ref_force = "pre_hold", F_MIN_REF
            elif t_norm < 0.8:
                frac = (t_norm - 0.2) / 0.6
                phase = "pressing_in" if frac <= 0.5 else "coming_out"
                ref_force = F_MIN_REF + (F_MAX_REF - F_MIN_REF) * np.sin(frac * np.pi)
            else:
                phase, ref_force = "post_hold", F_MIN_REF

        elif wf_type == "sigmoid":
            # 20% wait, 20% sigmoid up, 20% hold max, 20% sigmoid down, 20% wait
            if t_norm < 0.2:
                phase, ref_force = "pre_hold", F_MIN_REF
            elif t_norm < 0.4:
                phase = "pressing_in"
                frac = (t_norm - 0.2) / 0.2
                ref_force = F_MIN_REF + (F_MAX_REF - F_MIN_REF) * get_sigmoid_norm(frac)
            elif t_norm < 0.6:
                phase, ref_force = "holding", F_MAX_REF
            elif t_norm < 0.8:
                phase = "coming_out"
                frac = (t_norm - 0.6) / 0.2
                ref_force = F_MAX_REF - (F_MAX_REF - F_MIN_REF) * get_sigmoid_norm(frac)
            else:
                phase, ref_force = "post_hold", F_MIN_REF

        else:
            # Default fallback
            if t_norm <= 0.5:
                phase = "pressing_in"
                frac = t_norm / 0.5
                ref_force = F_MIN_REF + (F_MAX_REF - F_MIN_REF) * frac
            else:
                phase = "coming_out"
                frac = (t_norm - 0.5) / 0.5
                ref_force = F_MAX_REF - (F_MAX_REF - F_MIN_REF) * frac

        # Only reset PID when entering or leaving the non-contact hover phases
        if phase in ("pre_hold", "post_hold") and prev_phase not in ("pre_hold", "post_hold"):
            pid.reset()
        prev_phase = phase

        raw_force = get_normal_force(obj.id, digit_body.id)
        measured_force = raw_force * force_scale
        filtered_force = FORCE_FILTER_ALPHA * measured_force + (1.0 - FORCE_FILTER_ALPHA) * filtered_force

        # ---------------- HOVER RELEASE LOGIC ----------------
        if phase == "pre_hold":
            # Visually hover 1.5mm above the surface
            z_command = z_start + 0.0015
        elif phase == "post_hold":
            # Smoothly retract to the visual hover height to clear the gel
            z_command = min(z_command + (MAX_DELTA_Z * 2), z_start + 0.0015)
        else:
            # We are in an active ramp or hold phase.
            # If we are up in the air gap, snap down rapidly to the contact boundary
            if z_command > z_start + 0.0001:
                z_command = max(z_command - (MAX_DELTA_Z * 50), z_start)
            else:
                # Normal PID force control once physically touching the gel
                error = ref_force - filtered_force
                delta_z = pid.update(error, DT)
                z_command -= delta_z

        # Relative physical limits (allow up to 10mm compression below z_start and 1.5mm hover clearance)
        z_command = max(z_start - 0.010, min(z_command, z_start + 0.0015))

        p.resetBasePositionAndOrientation(obj.id, [LOCKED_X, LOCKED_Y, z_command], LOCKED_QUAT)
        p.stepSimulation()
        # time.sleep(DT)  # Commented out to unlock maximum speed

        if step % STEPS_PER_FRAME == 0:
            color, depth = digits.render()
            digits.updateGUI(color, depth)
            video_writer = write_video_frame(video_writer, color, video_path, VIDEO_PLAYBACK_FPS)

            # --- Frame Extraction ---
            frames_dir = out_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert color format for OpenCV and save using explicit frame_count
            frame = color[0] if isinstance(color, list) else color 
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            frame_path = frames_dir / f"frame_{frame_count:04d}.png"  # <--- Changed format here
            cv2.imwrite(str(frame_path), frame_bgr)
            
            frame_count += 1  # <--- Increment for the next extraction

        csv_writer.writerow(
            [
                f"{elapsed:.4f}",
                phase,
                f"{z_command:.6f}",
                f"{raw_force:.6f}",
                f"{ref_force:.4f}",
                f"{measured_force:.4f}",
                f"{filtered_force:.4f}",
            ]
        )
        times.append(elapsed)
        positions.append(z_command)
        measureds.append(filtered_force)
        references.append(ref_force)

        if step % print_interval_steps == 0:
            print(
                f"[{indenter_name}/{force_max_N:g}N/{duration_s:g}s] "
                f"t={elapsed:6.2f}s | Phase: {phase:12s} | Z: {z_command:.4f} m | "
                f"Target: {ref_force:5.2f} N | Measured: {measured_force:5.2f} N   ",
                end="\r",
            )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log.info("Interrupted by user.")
            break

    csv_file.close()
    if video_writer is not None:
        video_writer.release()
    p.stopStateLogging(gui_log_id)

    if len(times) > 0:
        plt.figure(figsize=(9, 5))
        plt.plot(times, references, label="Target Force (N)", linestyle="--", linewidth=2)
        plt.plot(times, measured, label="Measured Force (N)", linewidth=1.5)
        plt.axhline(F_MAX_REF, color="tab:blue", linestyle=":", linewidth=1, alpha=0.5, label=f"{F_MAX_REF:g}N limit")
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.title(f"{indenter_name.capitalize()} Indenter Force: {force_max_N:g}N, {duration_s:g}s, {waveform.capitalize()}")
        plt.legend()
        plt.grid(True)
        plot_path = out_dir / "force_ramp_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

    log.info(f"Experiment complete. Files written to: {out_dir}")


if __name__ == "__main__":
    main()
