"""Tactile force estimation CLI.

    python main.py index     [--config CFG] [key.sub=value ...]
    python main.py inspect
    python main.py checklabels   (do the force labels follow the contact visible in the frames?)
    python main.py train     --config configs/cnn.yaml      (or configs/vit.yaml)
    python main.py evaluate  --config configs/cnn.yaml [--ckpt PATH]
    python main.py predict   --config configs/cnn.yaml --input FRAME_OR_FOLDER [--reference NO_CONTACT_FRAME]
    python main.py compare   outputs/cnn outputs/vit
    python main.py all       --config configs/cnn.yaml      (index -> inspect -> train -> evaluate)

Unseen-shape experiment (edit configs/unseen_shapes.yaml first):
    python main.py train --config configs/cnn.yaml --config configs/unseen_shapes.yaml
    python main.py train --config configs/vit.yaml --config configs/unseen_shapes.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tactile_force.config import load_config  # noqa: E402
from tactile_force import pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["index", "inspect", "checklabels", "train", "evaluate", "predict", "compare", "all"])
    ap.add_argument("--config", action="append",
                    help="config file; repeat to combine, e.g. --config configs/cnn.yaml --config configs/unseen_shapes.yaml")
    ap.add_argument("--input", help="predict: a frame image or a frames folder")
    ap.add_argument("--reference", help="predict: no-contact frame for diff input modes")
    ap.add_argument("--ckpt", help="checkpoint (default: <work_dir>/best_model.pt)")
    ap.add_argument("--out", help="predict: output CSV path")
    ap.add_argument("overrides", nargs="*",
                    help="config overrides, e.g. train.epochs=5 data.root=D:/dataset (compare: output folders)")
    args = ap.parse_intermixed_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "compare":
        pipeline.cmd_compare(args.overrides, args.out)
        return
    cfg = load_config(args.config or ["configs/default.yaml"], args.overrides)

    if args.command == "index":
        pipeline.cmd_index(cfg)
    elif args.command == "inspect":
        pipeline.cmd_inspect(cfg)
    elif args.command == "checklabels":
        pipeline.cmd_check_labels(cfg)
    elif args.command == "train":
        pipeline.cmd_train(cfg)
    elif args.command == "evaluate":
        pipeline.cmd_evaluate(cfg, args.ckpt)
    elif args.command == "predict":
        if not args.input:
            ap.error("predict needs --input")
        pipeline.cmd_predict(cfg, args.input, args.reference, args.ckpt, args.out)
    elif args.command == "all":
        pipeline.cmd_index(cfg)
        pipeline.cmd_inspect(cfg)
        pipeline.cmd_train(cfg)


if __name__ == "__main__":  # required on Windows for DataLoader worker processes
    main()
