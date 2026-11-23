# Modules

## Entry points

| File | Purpose |
|---|---|
| `main.py` | Command-line interface of the learning pipeline: `index`, `inspect`, `checklabels`, `train`, `evaluate`, `predict`, `compare`, `all`. Config files with `--config` (repeatable) and overrides as `section.key=value`. |
| `run_split_experiment.py` | The strict unseen-indenter experiment (P1, exp000) in one command: checks the split, builds its index, trains and tests the CNN and the ViT on the same split, writes a comparison and `summary.json`. |
| `experiments/run_exp.py` | Numbered experiment runner (`expNNN_<name>`, never overwrites): shared index, shape split, training, validation and test metrics, one row appended to `outputs/experiments/results.csv`. Used for exp001–exp023. |

## Package `tactile_force/`

| Module | Contents |
|---|---|
| `config.py` | YAML loading with inheritance (`base: default.yaml`), merging of several files and dotted command-line overrides; `save_config` writes `config_used.yaml`. |
| `indexing.py` | Scans the dataset tree, parses shape / peak force / duration / profile from the path, finds the label CSV and aligns the force to every frame (frame column, render rows, time column or uniform rows; target-profile fallback). Writes `index.csv` and `coverage.txt`. |
| `splits.py` | Train/val/test splits grouped by run: `run` (stratified per shape), `shape` (whole indenters held out, `val_shapes`, `test_shapes`) and the few-shot calibration split (`calib_runs_per_force`, `calib_forces`, `calib_val_runs_per_force`) with leakage checks. Split files are cached as JSON. |
| `preprocess.py` | Image reading, optional top mask and crop, resize to 160 × 120, augmentation applied identically to frame and reference (horizontal/vertical flips, ±4 px shift, gain 0.9–1.1 and offset ±0.03, Gaussian noise σ = 0.01; optional `diff_gain_jitter`), and the network input: `raw`, `diff` or **`raw+diff`** (6 channels: frame and gain × (frame − reference)). The reference is the first frame of the run or the mean of the first `ref_frames` frames. |
| `data.py` | `FrameForceDataset` (PyTorch dataset over the index, cached run references, optional past-frame channels) and `subsample_per_run` (run-balanced sampling: at most `max_frames_per_run` frames per run and epoch, so 30 s runs do not outweigh 10 s runs). |
| `models.py` | `CNNRegressor` – 4 blocks of [3×3 conv–BN–ReLU]×2 + 2×2 max-pool (32/64/128/256 channels), global average pooling, dropout, FC 128, FC 1 (≈ 1.21 M parameters). `ViTRegressor` – 8×8 patch embedding (dim 192, 300 tokens) + [CLS] + learned position embedding, 6 pre-norm transformer blocks (6 heads, MLP ×4, GELU), LayerNorm, linear head (≈ 2.80 M parameters). `build_model(cfg)`. |
| `train.py` | Training loop: AdamW, one-cycle learning-rate schedule, Huber loss on force / 5 N, mixed precision on CUDA, run-balanced epochs, oversampling of calibration runs (`oversample_shapes`, `oversample_factor`), early stopping, `select: best | last`; saves `best_model.pt`, `last_model.pt`, `history.csv`. `load_model`, `predict_loader`. |
| `metrics.py` | MAE, RMSE, bias, R², 95th-percentile error and MAPE – overall, in contact (> `contact_threshold`) and without contact; grouped by shape, profile, peak force and duration. |
| `plots.py` | Parity plot, error vs force, measured vs predicted time series per run, training curves. |
| `pipeline.py` | Implementation of the CLI commands (`cmd_index`, `cmd_inspect`, `cmd_check_labels`, `cmd_train`, `cmd_evaluate`, `cmd_predict`, `cmd_compare`) and the evaluation output folder `eval_test/`. |
| `profiles.py` | The five commanded force profiles (`profile_shape`, `target_force`); used as fallback labels and by the generator. |

## Configuration files `configs/`

| File | Use |
|---|---|
| `default.yaml` | All settings with comments: data root and label CSV detection, preprocessing, split, model, training, evaluation. |
| `cnn.yaml` | CNN: lr 1e-3, 40 epochs, patience 8. |
| `vit.yaml` | ViT: lr 5e-4, weight decay 0.05, 60 epochs, patience 12. |
| `split_hex_hash.yaml` | P1 split: train 6 indenters, val hash, test hexagonal. |
| `unseen_shapes.yaml` | Template for other held-out indenters. |

## Experiment scripts `experiments/`

| Script | Purpose |
|---|---|
| `run_exp.py` | Numbered experiment (see above). Options `--name`, `--model cnn|vit`, `--val <shapes>|none`, `--test <shapes>`, `--note`, `--set key=value …`. |
| `ensemble_eval.py` | Flip test-time augmentation (TTA), seed ensembles and temporal smoothing (causal mean, centred median; window sizes) on top of trained experiments. All choices are made on the validation runs; caches predictions as parquet. |
| `smooth_study.py` | Post-hoc temporal smoothing of one experiment, window chosen on the validation runs. |
| `force_level_classify.py` | Force-level classification derived from a regressor: run level (which of 1–5 N) and frame level (no contact, 0.05–0.5 N, 1 N bands around 1–5 N); confusion matrices. |
| `shape_classify.py` | 8-class indenter-shape classifier with the exp021 backbone and input recipe; run-level 70/15/15 split; in-contact frames only (exp024). |
| `shape_classify_eval.py` | Metrics of the shape classifier from its saved predictions (accuracy, macro-F1, per shape, per force, confusion). |
| `feature_study.py` | Leave-one-shape-out regression on hand-crafted statistics of the difference image (why the strict task fails). |
| `oracle_norm_study.py` | Upper-bound check: per-run normalisation with an oracle 1 N level, leave-one-shape-out. |
| `plot_signal_vs_force.py` | Image statistics vs force per indenter (saturation plot). |
| `queue1.sh` … `queue11.sh` | The exact command sequences that produced exp001–exp023 (run from the repository root, e.g. inside tmux). |

## Phase 1 – data generation `data_generation/`

| File | Purpose |
|---|---|
| `generate_tacto_dataset.py` | Force-controlled TACTO/PyBullet generator of the released dataset: contact height, force–depth curve, feed-forward depth control, frames + labels + summary + plot + video per run, collection log per indenter. Hydra config `conf/digit.yaml`. |
| `profiles.py` | Copy of `tactile_force/profiles.py` imported by the generator. |
| `exploratory/demo_pybullet_digit_force.py` | Earlier generator with a PID loop on the simulated force (frames with depth panel and text overlay). |
| `exploratory/run_full_parameter_sweep.ps1` | PowerShell sweep of the exploratory generator: 8 indenters × 1–5 N × 10–30 s × 5 profiles. |
| `objects/indenter_template.urdf` | URDF template for an indenter STL. |

## Phase 2 – real rig `real_rig/`

| File | Purpose |
|---|---|
| `record_both_sensors.py` | Synchronised recording of the DIGIT (digit-interface) and the ESP32/HX711 load cell (pyserial): interactive tare/calibration, reader thread, common PC clock, per-frame load interpolation, CSV logs and PNG frames. |
| `config.yaml` | DIGIT serial/resolution/fps/LED, serial port/baud/calibration mass, output folder, interpolation. |

## Analysis and tools

| File | Purpose |
|---|---|
| `analysis/make_figures.py` | Produces the data figures of the thesis (dataset, results, error analysis) from the dataset and `outputs/`. Set `TACTO_DATA_ROOT`. |
| `analysis/make_training_overview.py` | Six-panel training-history figure of all experiments (`docs/images/training_all.png`). |
| `tools/make_synthetic_dataset.py` | Small fake dataset with the same layout, for installation checks and tests. |
| `tests/test_pipeline.py` | pytest suite (9 tests): both folder layouts, frame numbers, force profiles, run and shape splits without leakage, input shapes, one output per image for every architecture, config inheritance. |

## Released checkpoints `checkpoints/`

| File | Model |
|---|---|
| `exp021_cnn_force_final.pt` + `.yaml` | Final force CNN (P3 recipe, 30-frame mean reference, final epoch). Test on 115 hexagonal runs: 0.078 N contact MAE. |
| `exp024_cnn_shape_classifier.pt` | 8-class shape classifier (99.7 % frame accuracy). |

A force checkpoint is a dict with `model` (state dict), `cfg`, `epoch` and `val_mae_N`; load it with `tactile_force.train.load_model(path, cfg, device)`. The shape classifier stores `model`, `classes` and `epoch`.
