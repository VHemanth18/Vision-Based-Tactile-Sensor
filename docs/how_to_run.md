# How to run

All commands are run from the repository root. Set the dataset folder once:

```bash
export TACTO_DATA_ROOT=/path/to/TACTO_DATASETS          # Windows PowerShell: $env:TACTO_DATA_ROOT="D:\TACTO_DATASETS"
```

`main.py` and `run_split_experiment.py` take the folder as `data.root=…` / `--data-root …`;
`experiments/run_exp.py` and `analysis/make_figures.py` read `TACTO_DATA_ROOT`.

## 1. The `main.py` commands

| Command | What it does | Output |
|---|---|---|
| `index` | scan the dataset, attach the force label to every frame | `<index_dir>/index.csv`, `coverage.txt` |
| `inspect` | raw frame vs network input | `<index_dir>/inspect.png` |
| `checklabels` | do the labels follow the contact visible in the frames? | console report |
| `train` | train, keep the selected checkpoint, evaluate on the test split | `<work_dir>/best_model.pt`, `last_model.pt`, `history.csv`, `training_curves.png`, `eval_test/` |
| `evaluate` | re-evaluate a saved checkpoint (`--ckpt`) | `<work_dir>/eval_test/` |
| `predict` | force of one frame or of a whole `frames/` folder | CSV (+ PNG plot for a folder) |
| `compare` | several trained models in one table and bar chart | `comparison.csv/png` |
| `all` | index → inspect → train | |

```bash
# seen indenters, unseen runs (run-level split of all 8 indenters)
python main.py index --config configs/cnn.yaml data.root=$TACTO_DATA_ROOT
python main.py train --config configs/cnn.yaml data.root=$TACTO_DATA_ROOT
python main.py train --config configs/vit.yaml data.root=$TACTO_DATA_ROOT
python main.py compare outputs/cnn outputs/vit --out outputs/compare
```

Any setting can be overridden as `section.key=value`; lists need quotes in PowerShell (`"preprocess.size=[80,64]"`).

| Situation | Override |
|---|---|
| CPU only / slow | `"preprocess.size=[80,64]" train.max_frames_per_run=60` |
| DataLoader worker errors on Windows | `train.num_workers=0` |
| GPU out of memory | `train.batch_size=32` |
| Ablation: difference image only | `preprocess.input_mode=diff` |
| Mean reference over the first k frames | `preprocess.ref_frames=30` |
| Keep the final epoch instead of the best | `train.select=last` |

## 2. Strict unseen-indenter experiment (P1, exp000)

```bash
python run_split_experiment.py --data-root $TACTO_DATA_ROOT
```

Trains on ball_array, heart, ring, star, triangle, asterisk; early stopping on hash; tests on hexagonal; CNN and ViT on
the identical split. Writes `outputs/split_test-hexagonal_val-hash/` (`index/`, `cnn/`, `vit/`, `compare/`,
`summary.json`). Options: `--models cnn`, `--epochs N`, `--cpu-small`, `--num-workers N`, `--skip-index`,
`--train/--val/--test <shapes>` with a new `--out`, `--set key=value …`.

**Run this first**: its index (`outputs/split_test-hexagonal_val-hash/index/index.csv`) is the shared index of all
numbered experiments.

## 3. Numbered experiments (exp001 – exp023)

```bash
mkdir -p outputs/experiments/logs
python experiments/run_exp.py --name <name> --model cnn|vit [--val <shapes>|none] [--test <shapes>] [--note "..."] --set key=value ...
```

Each call creates the next free `outputs/experiments/expNNN_<name>/` (never overwrites) with `command.txt`,
`config_used.yaml`, checkpoints, `history.csv`, `training_curves.png`, `val_metrics.json`, `eval_test/`, and appends
one row to `outputs/experiments/results.csv`. On Windows add `--set train.num_workers=0`.

The commands of every experiment are in `experiments/queue1.sh` … `queue11.sh` (`bash experiments/queue6.sh`).
On a fresh clone the numbers start again at exp001, so compare by name. Overrides used:

| Experiment | `--model` | `--val` | `--set` |
|---|---|---|---|
| exp001 diff-only | cnn | hash | `preprocess.input_mode=diff` |
| exp002 + gain jitter | cnn | hash | `preprocess.input_mode=diff "preprocess.diff_gain_jitter=[0.6,1.6]"` |
| exp003 + avg/max pool | cnn | hash | exp002 + `model.cnn.pool=avgmax` |
| exp004 / exp012 seen indenters | cnn / vit | hash | `split.mode=run` |
| exp005 5 calib runs | cnn | hash | `split.calib_runs_per_force=1` |
| exp006 1 calib run at 3 N | cnn | hash | `split.calib_runs_per_force=1 "split.calib_forces=[3]"` |
| **exp007 few-shot calibration** | cnn | none | `CAL` (below) |
| exp008 / exp013 one run at 3 N / 5 N | cnn | none | `CAL "split.calib_forces=[3]"` / `[5]` |
| exp009 other split seed | cnn | none | `CAL split.seed=1` |
| exp010 heart | cnn | none | `CAL` with `train.oversample_shapes=[heart_indenter]` and `--test heart_indenter` |
| exp011 ViT | vit | none | `CAL` |
| exp014 80 epochs | cnn | none | `CAL train.epochs=80 train.early_stop_patience=15` |
| exp016 / exp017 init seeds | cnn | none | `CAL train.early_stop_patience=999 train.seed=1` / `2` |
| exp018 / exp020 past frames | cnn | none | `CAL train.early_stop_patience=999 "preprocess.temporal_offsets=[3,6]"` (+ `train.select=last`) |
| **exp021 final recipe** | cnn | none | `CAL train.early_stop_patience=999 train.select=last preprocess.ref_frames=30` |
| exp022 final recipe on heart | cnn | none | as exp021 with heart (`--test heart_indenter`) |
| exp023 final recipe, ViT | vit | none | as exp021 |

`CAL = split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 "train.oversample_shapes=[hexagonal_indenter]"`

Final recipe:

```bash
python experiments/run_exp.py --name cnn_cal5v5_hex_ref30_last --model cnn --val none --set \
  split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 \
  train.oversample_factor=10 "train.oversample_shapes=[hexagonal_indenter]" \
  train.early_stop_patience=999 train.select=last preprocess.ref_frames=30
```

On an RTX 5070 a CNN run of 40 epochs takes ≈ 40 min and a ViT run of 60 epochs ≈ 2.4 h.

## 4. Post-processing: TTA, ensembles, temporal smoothing

```bash
python experiments/ensemble_eval.py --out outputs/experiments/ensembles/exp021 \
    --exps outputs/experiments/exp021_cnn_cal5v5_hex_ref30_last
python experiments/ensemble_eval.py --out outputs/experiments/ensembles/ens_007_016_017 \
    --exps outputs/experiments/exp007_* outputs/experiments/exp016_* outputs/experiments/exp017_*
```

Evaluates flip TTA on/off and causal-mean / centred-median smoothing windows; every choice is made on the validation
runs and applied once to the test runs (`chosen_by_val.csv`, `grid.csv`). Recommended:

* real time: flip TTA + causal mean of 3 frames (0.1 s delay),
* offline: flip TTA + centred median of 25 frames (0.4 s look-ahead).

## 5. Force-level and shape classification

```bash
# force level from the regressor (no extra training): run level 1..5 N and frame-level bands
python experiments/force_level_classify.py outputs/experiments/exp021_cnn_cal5v5_hex_ref30_last

# 8-class shape classifier with the exp021 recipe (needs exp021's config_used.yaml and the shared index)
python experiments/shape_classify.py --epochs 20
python experiments/shape_classify_eval.py outputs/experiments/exp024_cnn_shape_classification
```

## 6. Prediction with the released model

```bash
python main.py predict --config checkpoints/exp021_cnn_force_final.yaml --ckpt checkpoints/exp021_cnn_force_final.pt \
    --input $TACTO_DATA_ROOT/hexagonal_indenter/force_3N/duration_20s/sigmoid/frames --out pred.csv
# single frame: give the no-contact reference frame
python main.py predict --config checkpoints/exp021_cnn_force_final.yaml --ckpt checkpoints/exp021_cnn_force_final.pt \
    --input .../frames/frame_00300.png --reference .../frames/frame_00001.png
```

For a folder, the first frame is used as the no-contact reference, so the folder must start without contact. The
released model is calibrated for the eight simulated indenters (the hexagonal one with 5 calibration runs). For a new
indenter or for real frames, calibrate it first with a few labelled runs (protocol P3).

In Python:

```python
from tactile_force.config import load_config
from tactile_force.train import load_model, pick_device
cfg = load_config(["checkpoints/exp021_cnn_force_final.yaml"])
model = load_model("checkpoints/exp021_cnn_force_final.pt", cfg, pick_device("auto"))
```

## 7. Analysis studies and figures

```bash
python experiments/feature_study.py --index outputs/split_test-hexagonal_val-hash/index/index.csv --out outputs/analysis/features.parquet
python experiments/plot_signal_vs_force.py outputs/analysis/features.parquet outputs/analysis/signal_vs_force.png
python experiments/oracle_norm_study.py outputs/analysis/features.parquet
TACTO_DATA_ROOT=/path/to/TACTO_DATASETS python analysis/make_figures.py        # -> outputs/figures/*.pdf
python analysis/make_training_overview.py                                        # -> outputs/figures/training_all.png
```

`make_figures.py` and `make_training_overview.py` read the experiment folders under `outputs/` (as produced by the
commands above; `outputs/archive/experiments/` for older experiments).

## 8. Dataset generation and real-world recording

* Simulated dataset: [dataset_preparation.md](dataset_preparation.md).
* Real rig: [phase2_real_rig.md](phase2_real_rig.md).

## 9. Tests

```bash
python -m pytest -q
```
