"""Unseen-shape experiment with a fixed shape split: train CNN and ViT, test on a held-out shape.

    Train      : ball_array, heart, ring, star, triangle, asterisk
    Validation : hash        (early stopping and best-checkpoint selection)
    Test       : hexagonal   (final evaluation; never seen during training)

Run from the repository root (the folder that contains main.py):

    python run_split_experiment.py --data-root /path/to/TACTO_DATASETS
    python run_split_experiment.py --data-root /path/to/TACTO_DATASETS --models cnn --epochs 20 --cpu-small
    python run_split_experiment.py --data-root /content/work/data/DS --num-workers 2          # Colab

Outputs (default outputs/split_test-hexagonal_val-hash/):
    index/            index.csv, coverage.txt and the split file for this experiment only
    cnn/, vit/        best_model.pt, history.csv, training_curves.png, eval_test/ (metrics on hexagonal)
    compare/          comparison.csv / comparison.png
    summary.json      shapes per split and the headline test metrics of each model
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tactile_force import pipeline  # noqa: E402
from tactile_force.config import load_config  # noqa: E402

TRAIN_SHAPES = ["ball_array_indenter", "heart_indenter", "ring_indenter",
                "star_indenter", "triangle_indenter", "asterisk_indenter"]
VAL_SHAPES = ["hash_indenter"]
TEST_SHAPES = ["hexagonal_indenter"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder that contains the <shape>_indenter folders")
    ap.add_argument("--out", default="outputs/split_test-hexagonal_val-hash", help="experiment output folder")
    ap.add_argument("--models", nargs="+", default=["cnn", "vit"], choices=["cnn", "vit"])
    ap.add_argument("--train", nargs="+", default=TRAIN_SHAPES, help="training shapes (folder names)")
    ap.add_argument("--val", nargs="+", default=VAL_SHAPES, help="validation shapes")
    ap.add_argument("--test", nargs="+", default=TEST_SHAPES, help="test shapes")
    ap.add_argument("--epochs", type=int, help="override epochs for every model (default: configs/<model>.yaml)")
    ap.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 2,
                    help="DataLoader workers (default 0 on Windows, 2 elsewhere)")
    ap.add_argument("--cpu-small", action="store_true",
                    help="CPU-friendly: 80x64 input and 60 frames per run per epoch (same for both models)")
    ap.add_argument("--skip-index", action="store_true", help="reuse <out>/index/index.csv")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="extra config overrides, e.g. train.lr=0.0003 train.batch_size=32")
    return ap.parse_args()


def check_shapes(data_root: Path, train: list[str], val: list[str], test: list[str]) -> None:
    for a_name, a, b_name, b in (("train", train, "val", val), ("train", train, "test", test),
                                 ("val", val, "test", test)):
        overlap = set(a) & set(b)
        if overlap:
            sys.exit(f"Shapes in both {a_name} and {b_name}: {sorted(overlap)}")
    if not data_root.is_dir():
        sys.exit(f"--data-root {data_root} does not exist")
    present = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    missing = [s for s in train + val + test if s not in present]
    if missing:
        sys.exit(f"Shape folders not found in {data_root}: {missing}\nFound: {present}")
    ignored = [s for s in present if s not in train + val + test]
    if ignored:
        print(f"Not used in this experiment: {ignored}")


def build_cfg(model: str, args: argparse.Namespace, out: Path):
    lst = lambda xs: "[" + ",".join(xs) + "]"  # noqa: E731
    overrides = [
        f"data.root={Path(args.data_root).as_posix()}",
        f"index_dir={(out / 'index').as_posix()}",      # own index: never overwrites outputs/index
        f"work_dir={(out / model).as_posix()}",
        f"data.include_shapes={lst(args.train + args.val + args.test)}",
        "data.exclude_shapes=[]",
        "split.mode=shape",
        f"split.test_shapes={lst(args.test)}",
        f"split.val_shapes={lst(args.val)}",
        f"train.num_workers={args.num_workers}",
    ]
    if args.epochs:
        overrides.append(f"train.epochs={args.epochs}")
    if args.cpu_small:
        overrides += ["preprocess.size=[80,64]", "train.max_frames_per_run=60"]
    overrides += args.set
    return load_config([ROOT / "configs" / f"{model}.yaml"], overrides)


def verify_split(cfg, args: argparse.Namespace) -> None:
    """Stop before training if any split contains a shape it should not."""
    parts = pipeline._splits(cfg)
    expected = {"train": set(args.train), "val": set(args.val), "test": set(args.test)}
    for name, frames in parts.items():
        got = set(frames["shape"].unique())
        if got != expected[name]:
            sys.exit(f"{name} split has shapes {sorted(got)}, expected {sorted(expected[name])}. "
                     f"Check {cfg.index_dir}/coverage.txt for shapes without labelled runs.")


def headline(work_dir: Path) -> dict:
    path = work_dir / "eval_test" / "metrics.json"
    if not path.exists():
        return {}
    rep = json.loads(path.read_text())
    return {"overall": rep.get("overall", {}), "in_contact": rep.get("in_contact", {})}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_shapes(Path(args.data_root), args.train, args.val, args.test)

    cfgs = {m: build_cfg(m, args, out) for m in args.models}
    first = cfgs[args.models[0]]
    if args.skip_index and (out / "index" / "index.csv").exists():
        print(f"Reusing {out / 'index' / 'index.csv'}")
    else:
        pipeline.cmd_index(first)          # one index shared by every model -> identical splits
    verify_split(first, args)

    for model, cfg in cfgs.items():
        print(f"\n===== {model.upper()}: train on {len(args.train)} shapes, "
              f"val={args.val}, test={args.test} =====", flush=True)
        pipeline.cmd_train(cfg)            # trains, keeps best_model.pt, evaluates on the test shapes

    dirs = [str(out / m) for m in args.models]
    if len(dirs) > 1:
        pipeline.cmd_compare(dirs, str(out / "compare"))

    summary = {"data_root": str(args.data_root), "train": args.train, "val": args.val, "test": args.test,
               "results": {m: headline(out / m) for m in args.models}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTest-shape results (in-contact frames):")
    for m, res in summary["results"].items():
        c = res.get("in_contact", {})
        if c:
            print(f"  {m.upper():4s} MAE {c['mae_N']:.3f} N | RMSE {c['rmse_N']:.3f} N | "
                  f"bias {c['bias_N']:+.3f} N | R2 {c['r2']:.3f}")
    print(f"Wrote {out / 'summary.json'}")


if __name__ == "__main__":  # required on Windows for DataLoader worker processes
    main()
