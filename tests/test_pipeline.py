"""Fast checks: path parsing, split leakage, preprocessing shapes.  Run: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tactile_force.config import load_config  # noqa: E402
from tactile_force.indexing import frame_number, parse_run  # noqa: E402
from tactile_force.preprocess import crop_and_resize, in_channels, to_input  # noqa: E402
from tactile_force.profiles import target_force  # noqa: E402
from tactile_force.splits import apply_split, make_splits  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
PROFILES = ["square", "trapezoidal", "triangular", "sinusoidal", "sigmoid"]


def test_parse_both_layouts(tmp_path):
    a = tmp_path / "heart_indenter/force_3N/duration_25s/sigmoid/frames"
    b = tmp_path / "sphere/trapezoidal/2N_15s/frames"
    for p in (a, b):
        p.mkdir(parents=True)
    ra, rb = parse_run(a, tmp_path, PROFILES), parse_run(b, tmp_path, PROFILES)
    assert (ra.shape, ra.peak_force, ra.duration, ra.profile) == ("heart_indenter", 3.0, 25.0, "sigmoid")
    assert (rb.shape, rb.peak_force, rb.duration, rb.profile) == ("sphere", 2.0, 15.0, "trapezoidal")


def test_frame_number():
    assert frame_number(Path("frame_00042.png")) == 42
    assert frame_number(Path("img7_000123.jpg")) == 123


def test_profiles_peak_and_rest():
    t = np.linspace(0, 10, 1001)
    for k in PROFILES:
        f = target_force(k, 4.0, 10.0, t)
        assert abs(f.max() - 4.0) < 0.05 and f[t < 2].max() == 0 and f[t > 8].max() == 0


def test_run_split_has_no_leakage():
    cfg = load_config(CFG)
    rows = [{"run_id": f"s{s}|p|{f}N|10s", "shape": f"s{s}"} for s in range(3) for f in range(10) for _ in range(5)]
    idx = pd.DataFrame(rows)
    parts = apply_split(idx, make_splits(idx, cfg))
    sets = [set(parts[k]["run_id"]) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert all(len(parts[k]["shape"].unique()) == 3 for k in ("train", "val", "test"))


def test_input_shapes_two_panel_frames():
    cfg = load_config(CFG, ["preprocess.mask_top_px=22", "preprocess.crop=[0,0.5,0,1]"])
    frame = np.random.randint(1, 255, (160, 240, 3), np.uint8)
    img = crop_and_resize(frame, cfg.preprocess)
    assert img.shape == (160, 120, 3) and img[:22].max() == 0  # text band masked
    x = to_input(img, img, cfg.preprocess)
    assert x.shape == (in_channels(cfg.preprocess), 160, 120)


def test_input_shapes_single_tactile_frame():
    cfg = load_config(CFG)  # default: whole frame, no mask
    frame = np.random.randint(1, 255, (160, 120, 3), np.uint8)
    img = crop_and_resize(frame, cfg.preprocess)
    assert img.shape == (160, 120, 3) and img.min() > 0
    assert np.allclose(img, frame / 255.0, atol=1e-6)  # untouched


def test_all_architectures_output_one_value_per_image():
    import torch

    from tactile_force.models import ARCHS, build_model

    for arch in ARCHS:
        cfg = load_config(CFG, [f"model.arch={arch}"])
        y = build_model(cfg, in_channels(cfg.preprocess))(torch.randn(2, 6, 160, 120))
        assert y.shape == (2,), arch


def test_config_inheritance():
    cfg = load_config(CFG.parent / "vit.yaml", ["train.epochs=3"])
    assert cfg.model.arch == "vit" and cfg.train.epochs == 3 and cfg.index_dir == "outputs/index"


def test_shape_split_holds_out_whole_shapes():
    cfg = load_config([CFG, CFG.parent / "unseen_shapes.yaml"],
                      ["split.test_shapes=[s7,s8,s9]", "split.val_shapes=[s6]"])
    rows = [{"run_id": f"s{s}|p|{f}N|10s", "shape": f"s{s}", "peak_force": float(f)}
            for s in range(10) for f in range(1, 6)]
    parts = apply_split(pd.DataFrame(rows), make_splits(pd.DataFrame(rows), cfg))
    assert set(parts["test"]["shape"]) == {"s7", "s8", "s9"}
    assert set(parts["val"]["shape"]) == {"s6"}
    assert set(parts["train"]["shape"]) == {f"s{i}" for i in range(6)}
    assert cfg.work_dir == "outputs/unseen_shapes/cnn"
