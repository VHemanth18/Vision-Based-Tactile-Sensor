"""Run one numbered experiment: train -> evaluate on val and test -> append to results.csv.

Every run gets the next free number (exp001_<name>, exp002_<name>, ...) under outputs/experiments/;
an existing experiment folder is never reused or overwritten.

    python experiments/run_exp.py --name cnn_diff --model cnn --set preprocess.input_mode=diff

Default split = the main experiment: train on 6 shapes, val = hash, test = hexagonal.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tactile_force import pipeline  # noqa: E402
from tactile_force.config import load_config  # noqa: E402

EXP_DIR = ROOT / "outputs" / "experiments"
INDEX_DIR = EXP_DIR / "_index"
ARCHIVE_DIR = ROOT / "outputs" / "archive" / "experiments"
BASE_INDEX = ROOT / "outputs" / "split_test-hexagonal_val-hash" / "index" / "index.csv"
DATA_ROOT = os.environ.get("TACTO_DATA_ROOT", "/path/to/TACTO_DATASETS")   # set TACTO_DATA_ROOT to your dataset folder
TRAIN = ["ball_array_indenter", "heart_indenter", "ring_indenter", "star_indenter", "triangle_indenter",
         "asterisk_indenter"]


def next_exp_dir(name: str) -> Path:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    dirs = list(EXP_DIR.iterdir()) + list((ARCHIVE_DIR).glob("exp*"))  # archived experiments keep their numbers
    nums = [int(m.group(1)) for p in dirs if (m := re.match(r"exp(\d{3})_", p.name))]
    d = EXP_DIR / f"exp{max(nums, default=0) + 1:03d}_{name}"
    d.mkdir()  # fails if it exists: never overwrite an experiment
    return d


def shared_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    dst = INDEX_DIR / "index.csv"
    if not dst.exists():
        shutil.copy2(BASE_INDEX, dst)
        dst.chmod(0o444)


def lst(xs):
    return "[" + ",".join(xs) + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="cnn", choices=["cnn", "vit"])
    ap.add_argument("--val", nargs="+", default=["hash_indenter"])
    ap.add_argument("--test", nargs="+", default=["hexagonal_indenter"])
    ap.add_argument("--note", default="")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    shared_index()
    wd = next_exp_dir(a.name)
    all_shapes = sorted(set(TRAIN + ["hash_indenter", "hexagonal_indenter"]))
    overrides = [f"data.root={DATA_ROOT}", f"index_dir={INDEX_DIR.as_posix()}", f"work_dir={wd.as_posix()}",
                 f"data.include_shapes={lst(all_shapes)}", "split.mode=shape", f"split.test_shapes={lst(a.test)}",
                 f"split.val_shapes={lst([] if a.val == ['none'] else a.val)}", "train.num_workers=8"] + a.set
    cfg = load_config([ROOT / "configs" / f"{a.model}.yaml"], overrides)
    (wd / "command.txt").write_text(" ".join(sys.argv) + "\n")
    print(f"\n######## {wd.name} ########  model={a.model}  overrides={a.set}  {a.note}", flush=True)

    t0 = time.time()
    pipeline.cmd_train(cfg)  # train + evaluate on test -> eval_test/
    parts = pipeline._splits(cfg)
    tr, va_, te = (set(parts[k]["run_id"]) for k in ("train", "val", "test"))
    assert not (tr & te or va_ & te or tr & va_), "split leakage between train/val/test"

    # validation-shape metrics of the best checkpoint (model selection without the test shape)
    from tactile_force.metrics import full_report
    from tactile_force.train import load_model, make_loader, pick_device, predict_loader

    dev = pick_device(cfg.train.device)
    model = load_model(wd / "best_model.pt", cfg, dev)
    va = parts["val"].iloc[::3].reset_index(drop=True)
    va["pred"] = predict_loader(model, make_loader(va, cfg, train=False, batch_size=256), dev,
                                cfg.train.force_scale)
    vrep = full_report(va, cfg.evaluate.contact_threshold)
    (wd / "val_metrics.json").write_text(json.dumps(vrep, indent=2))

    trep = json.loads((wd / "eval_test" / "metrics.json").read_text())
    import pandas as pd

    hist = pd.read_csv(wd / "history.csv")
    row = {"exp": wd.name, "model": a.model, "val": "+".join(a.val), "test": "+".join(a.test),
           "overrides": " ".join(a.set), "note": a.note, "epochs_run": len(hist),
           "best_epoch": int(hist.loc[hist["val_mae_N"].idxmin(), "epoch"]),
           "val_contact_mae": vrep["in_contact"]["mae_N"], "val_all_mae": vrep["overall"]["mae_N"],
           "test_contact_mae": trep["in_contact"]["mae_N"], "test_contact_rmse": trep["in_contact"]["rmse_N"],
           "test_contact_bias": trep["in_contact"]["bias_N"], "test_contact_r2": trep["in_contact"]["r2"],
           "test_all_mae": trep["overall"]["mae_N"], "test_all_r2": trep["overall"]["r2"],
           "minutes": round((time.time() - t0) / 60, 1)}
    res = EXP_DIR / "results.csv"
    pd.DataFrame([row]).to_csv(res, mode="a", header=not res.exists(), index=False)
    print("RESULT " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
          flush=True)


if __name__ == "__main__":
    main()
