"""Scan the dataset tree and build a per-frame index with force labels.

Supported layouts (metadata is parsed from the path relative to data.root):
    <shape>/force_<F>N/duration_<T>s/<profile>/frames/*.png     (your current tree)
    <shape>/<profile>/<F>N_<T>s/frames/*.png                    (demo_pybullet_digit_force.py)

Label alignment per run, in order of preference:
    1. CSV has a frame column              -> join on frame index
    2. CSV has render rows (gel_depth_mm)  -> n-th rendered row = video frame n
    3. CSV has a time column               -> interpolate force at t = (frame_idx - first_idx) / fps
       (first_idx = lowest frame number of the run, so 0- and 1-based numbering both map to t = 0)
    4. no CSV (and fallback allowed)       -> commanded target profile at t
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .profiles import target_force

log = logging.getLogger(__name__)

_FORCE_RE = re.compile(r"^(?:force_)?(\d+(?:\.\d+)?)N", re.IGNORECASE)
_DUR_RE = re.compile(r"(?:^duration_|_)(\d+(?:\.\d+)?)s$", re.IGNORECASE)
_INT_RE = re.compile(r"(\d+)(?!.*\d)")

TIME_COLS = ["time_s", "time", "t", "timestamp"]
FORCE_COLS = ["measured_force_n", "measured_force", "force_n", "force", "filtered_force_n"]
TARGET_COLS = ["target_force_n", "target_force", "ref_force_n"]
FRAME_COLS = ["frame", "frame_idx", "frame_index", "frame_id"]
RENDER_COLS = ["gel_depth_mm"]


@dataclass
class RunMeta:
    run_dir: Path
    frames_dir: Path
    shape: str
    peak_force: float
    duration: float
    profile: str

    @property
    def run_id(self) -> str:
        return f"{self.shape}|{self.profile}|{self.peak_force:g}N|{self.duration:g}s"


def parse_run(frames_dir: Path, root: Path, profiles: list[str]) -> RunMeta | None:
    run_dir = frames_dir.parent
    parts = run_dir.relative_to(root).parts
    if not parts:
        return None
    shape, force, duration, profile = parts[0], None, None, None
    for part in parts[1:]:
        low = part.lower()
        if low in profiles:
            profile = low
        m = _FORCE_RE.match(part)
        if m:
            force = float(m.group(1))
        m = _DUR_RE.search(part)
        if m:
            duration = float(m.group(1))
    if None in (force, duration, profile):
        log.warning("Cannot parse shape/force/duration/profile from %s; skipped", run_dir)
        return None
    return RunMeta(run_dir, frames_dir, shape, force, duration, profile)


def frame_number(path: Path) -> int:
    m = _INT_RE.search(path.stem)
    if not m:
        raise ValueError(f"No frame number in file name: {path.name}")
    return int(m.group(1))


def _find_csv(run: RunMeta, patterns: list[str]) -> Path | None:
    for folder in (run.run_dir, run.frames_dir):
        for pat in patterns:
            hits = sorted(folder.glob(pat))
            if hits:
                return hits[0]
    return None


_NOT_MEASURED = ("target", "ref", "cmd", "command", "desired", "setpoint", "set_point", "goal")


def _fuzzy_force_column(df: pd.DataFrame) -> str | None:
    """Fallback: a numeric column whose name contains 'force' and is not a commanded/target value."""
    cands = [c for c in df.columns if "force" in c.lower() and not any(k in c.lower() for k in _NOT_MEASURED)
             and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    if not cands:
        return None
    for key in ("meas", "sensor", "actual", "filt", "contact"):
        for c in cands:
            if key in c.lower():
                return c
    return cands[0]


def _fuzzy_time_column(df: pd.DataFrame, duration: float) -> tuple[str, float] | None:
    """Fallback: an increasing numeric column whose name looks like time. Returns (column, scale to seconds)."""
    for c in df.columns:
        low = c.lower().replace(" ", "")
        if not ("time" in low or low in ("t", "t_s", "t(s)", "sec", "secs", "seconds", "elapsed", "elapsed_s")):
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() < 0.9 or not v.dropna().is_monotonic_increasing:
            continue
        span = float(v.max() - v.min())
        if span <= 0:
            continue
        # milliseconds or microseconds instead of seconds?
        for scale in (1.0, 1e-3, 1e-6):
            if 0.5 * duration <= span * scale <= 2.0 * duration:
                return c, scale
    return None


def _pick(df: pd.DataFrame, candidates: list[str], explicit: str | None = None) -> str | None:
    if explicit and explicit != "auto":
        if explicit not in df.columns:
            raise ValueError(f"column {explicit!r} not in CSV columns {list(df.columns)}")
        return explicit
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _labels_for_run(run: RunMeta, frame_idx: np.ndarray, first_idx: int, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Returns (time_s, force, target_force, label_source, fps)."""
    d = cfg.data
    span = max(int(frame_idx.max()) - first_idx, 1)
    fps = float(d.fps) if d.fps != "auto" else span / run.duration
    t = (frame_idx - first_idx) / fps
    tgt = target_force(run.profile, run.peak_force, run.duration, t, d.pre_rest, d.post_rest)

    csv_path = _find_csv(run, d.label_csv_patterns)
    if csv_path is not None:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # empty / corrupt file
            raise ValueError(f"cannot read {csv_path.name}: {type(exc).__name__}") from exc
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty:
            raise ValueError(f"{csv_path.name} has no rows")
        fcol = _pick(df, FORCE_COLS, d.force_column) or _fuzzy_force_column(df)
        if fcol is None:
            raise ValueError(f"no force column; CSV columns are {list(df.columns)}; set data.force_column")
        tcol, frcol = _pick(df, TIME_COLS), _pick(df, FRAME_COLS)
        rcol, gcol = _pick(df, RENDER_COLS), _pick(df, TARGET_COLS)

        if frcol is not None:
            sub = df.set_index(df[frcol].astype(int))
            sub = sub[~sub.index.duplicated()]
            force = sub[fcol].reindex(frame_idx).to_numpy(dtype=float)
            if gcol:
                tgt = sub[gcol].reindex(frame_idx).to_numpy(dtype=float)
            if tcol:
                t = sub[tcol].reindex(frame_idx).to_numpy(dtype=float)
            return t, force, tgt, "csv_frame", fps

        if rcol is not None:
            rendered = df[pd.to_numeric(df[rcol], errors="coerce").notna()].reset_index(drop=True)
            if len(rendered) > frame_idx.max():
                rows = rendered.iloc[frame_idx]
                force = rows[fcol].to_numpy(dtype=float)
                if gcol:
                    tgt = rows[gcol].to_numpy(dtype=float)
                if tcol:
                    t = rows[tcol].to_numpy(dtype=float)
                return t, force, tgt, "csv_render_rows", fps

        scale, source = 1.0, "csv_time"
        if tcol is None:
            fuzzy = _fuzzy_time_column(df, run.duration)
            if fuzzy:
                (tcol, scale), source = fuzzy, "csv_time_fuzzy"
        if tcol is not None:
            ts = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=float) * scale
            ts = ts - np.nanmin(ts)  # time relative to the start of the run
        elif d.get("assume_uniform_rows", True):
            # No time column: assume the rows were logged at a constant rate over the whole run.
            ts, source = np.linspace(0.0, run.duration, len(df)), "csv_uniform_rows"
        else:
            raise ValueError(f"cannot align rows to frames: no frame/time column; CSV columns are "
                             f"{list(df.columns)}")
        force = np.interp(t, ts, pd.to_numeric(df[fcol], errors="coerce").to_numpy(dtype=float))
        if gcol:
            tgt = np.interp(t, ts, pd.to_numeric(df[gcol], errors="coerce").to_numpy(dtype=float))
        return t, force, tgt, source, fps

    if not d.allow_target_fallback:
        raise FileNotFoundError("no label CSV in the run folder")
    return t, tgt.copy(), tgt, "target_profile", fps


def build_index(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (frame_index, coverage_report)."""
    d = cfg.data
    root = Path(d.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data.root does not exist: {root}")
    profiles = [p.lower() for p in d.profiles]
    exts = {e.lower() for e in d.image_exts}

    # One directory walk: collect frames folders without listing their images, and remember
    # profile folders that have no frames folder.
    t_start = time.time()
    print(f"Scanning {root} ...", flush=True)
    frames_dirs, no_frames = [], []
    for dirpath, dirnames, _ in os.walk(root):
        if d.frames_dirname in dirnames:
            frames_dirs.append(Path(dirpath) / d.frames_dirname)
            dirnames.remove(d.frames_dirname)  # do not descend into the image folder
        elif Path(dirpath).name.lower() in profiles:
            no_frames.append(Path(dirpath) / d.frames_dirname)
    frames_dirs.sort()
    print(f"Found {len(frames_dirs)} frames folders; reading frames and labels ...", flush=True)
    if not frames_dirs:
        print(f"No '{d.frames_dirname}' folders under {root}. Expected <shape>/force_XN/duration_Ys/<profile>/"
              f"{d.frames_dirname}/", flush=True)

    excluded = set(d.get("exclude_shapes") or [])
    included = set(d.get("include_shapes") or [])
    if excluded or included:
        print(f"Shape filter: include={sorted(included) or 'all'} exclude={sorted(excluded) or 'none'}", flush=True)
    rows, report = [], []
    for k, fdir in enumerate(frames_dirs, 1):
        if k % 25 == 0 or k == len(frames_dirs):
            print(f"  {k}/{len(frames_dirs)} runs  ({time.time() - t_start:.0f}s)", flush=True)
        run = parse_run(fdir, root, profiles)
        if run is None:
            continue
        if run.shape in excluded or (included and run.shape not in included):
            continue
        with os.scandir(fdir) as it:
            all_files = sorted((Path(e.path) for e in it
                                if e.is_file() and os.path.splitext(e.name)[1].lower() in exts), key=frame_number)
        if not all_files:
            report.append({"run_id": run.run_id, "shape": run.shape, "status": "empty_frames_dir"})
            continue
        ref = all_files[0]  # reference frame (no contact) = lowest-numbered frame, regardless of stride
        files = all_files[:: max(1, int(d.frame_stride))]
        idx = np.array([frame_number(f) for f in files])
        try:
            t, force, tgt, source, fps = _labels_for_run(run, idx, frame_number(ref), cfg)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            log.warning("Skipping %s: %s", run.run_id, exc)
            report.append({"run_id": run.run_id, "shape": run.shape, "status": "label_error",
                           "error": str(exc).replace(str(root), "<root>")})
            continue
        ok = np.isfinite(force)
        report.append({"run_id": run.run_id, "shape": run.shape, "status": "ok", "label_source": source,
                       "n_frames": int(ok.sum()), "fps": round(fps, 3),
                       "ref_force": float(force[0]) if ok[0] else np.nan})
        for f, i, ti, fi, gi in zip(np.array(files)[ok], idx[ok], t[ok], force[ok], tgt[ok]):
            rows.append((str(f), str(ref), run.run_id, run.shape, run.profile, run.peak_force,
                         run.duration, int(i), float(ti), float(fi), float(gi), source))

    have = {r["run_id"] for r in report}
    for fdir in no_frames:
        run = parse_run(fdir, root, profiles)
        if run and (run.shape in excluded or (included and run.shape not in included)):
            continue
        if run and run.run_id not in have:
            report.append({"run_id": run.run_id, "shape": run.shape, "status": "no_frames_dir"})
    print(f"Done in {time.time() - t_start:.0f}s", flush=True)

    cols = ["path", "ref_path", "run_id", "shape", "profile", "peak_force", "duration",
            "frame_idx", "time_s", "force", "target_force", "label_source"]
    index = pd.DataFrame(rows, columns=cols)
    return index, pd.DataFrame(report)


def summarize(index: pd.DataFrame, report: pd.DataFrame) -> str:
    if report.empty:
        return "No runs found: check data.root and the folder layout."
    lines = [f"Frames indexed: {len(index):,}   runs with frames: {index['run_id'].nunique()}"]
    status = report.groupby(["shape", "status"]).size().unstack(fill_value=0)
    lines.append("\nRuns per shape by status:\n" + status.to_string())
    if "error" in report.columns and report["error"].notna().any():
        err = report.dropna(subset=["error"]).copy()
        err["error"] = err["error"].str.replace(r"<root>[^:]*?(?=:|$)", "<run>", regex=True).str[:200]
        lines.append("\nLabel errors (count per shape and message):")
        for (shape, msg), n in err.groupby(["shape", "error"]).size().items():
            lines.append(f"  {shape:28s} x{n:<4d} {msg}")
        lines.append("Example run with an error: " + err.iloc[0]["run_id"])
    if len(index):
        src = index.groupby("label_source")["run_id"].nunique()
        lines.append("\nLabel source (runs):\n" + src.to_string())
        if (src.index == "csv_uniform_rows").any():
            lines.append("NOTE: csv_uniform_rows = the CSV has no time column; rows are assumed to be evenly spaced "
                         "over the run. Check those runs with `inspect` / the run plots.")
        if (src.index == "target_profile").any():
            lines.append("WARNING: some runs use the commanded target profile as label (no CSV found).")
        bad_ref = report[(report["status"] == "ok") & (report["ref_force"].abs() > 0.05)]
        if len(bad_ref):
            lines.append(f"WARNING: {len(bad_ref)} runs have force > 0.05 N in their first frame; "
                         "diff input modes assume the first frame is contact-free.")
        lines.append(f"\nForce label range: {index['force'].min():.3f} .. {index['force'].max():.3f} N")
    return "\n".join(lines)
