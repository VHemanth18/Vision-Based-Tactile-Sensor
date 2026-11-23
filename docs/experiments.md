# Experiments

Every experiment has its own folder in `results/experiments/expNNN_<name>/` with `command.txt`, `config_used.yaml`,
`history.csv`, `training_curves.png`, `val_metrics.json` and `eval_test/` (`metrics.json`, metrics by shape, profile,
peak force and duration, `parity.png`, `error_vs_force.png`, example runs). One row per experiment is in
`results/results_exp001-022.csv` and `results/results_exp023.csv`; the original log is `results/EXPERIMENTS.md`;
training logs are in `results/logs/`. Checkpoints and per-frame predictions are not included (except the two
released models in `checkpoints/`).

Dataset: 8 indenters × 125 runs, 601,000 frames. Metrics on the TEST runs; **contact** = frames with force > 0.05 N.
Hardware: RTX 5070, PyTorch 2.11.

## Force regression

| exp | model | setting | test | contact MAE (N) | contact RMSE | bias | contact R² | all-frame MAE | all R² |
|---|---|---|---|---|---|---|---|---|---|
| 000 | CNN | P1 strict unseen indenter (val hash) | hexagonal (125 runs) | 2.014 | 2.149 | +2.014 | −1.118 | 1.178 | −0.135 |
| 000 | ViT | P1 strict unseen indenter (val hash) | hexagonal | 2.721 | 3.132 | +2.721 | −3.500 | 1.586 | −1.412 |
| 001 | CNN | strict, difference image only | hexagonal | 3.282 | 3.433 | +3.280 | −4.408 | 1.913 | −1.898 |
| 002 | CNN | strict, diff + gain jitter | hexagonal | 3.133 | 3.265 | +3.133 | −3.890 | 1.832 | −1.629 |
| 003 | CNN | strict, diff + jitter + avg/max pool | hexagonal | 3.804 | 4.006 | +3.804 | −6.363 | 2.223 | −2.946 |
| — | constant | predict 1.96 N (train median) on contact frames | hexagonal | 1.237 | | | | | |
| 004 | CNN | P2 seen indenters, unseen runs (upper bound) | runs of all 8 indenters | 0.083 | 0.121 | −0.005 | 0.993 | 0.048 | 0.996 |
| 005 | CNN | 5 hex calib runs, early stop on hash | 120 hex runs | 0.882 | 1.191 | −0.854 | 0.348 | 0.527 | 0.651 |
| 006 | CNN | 1 hex calib run at 3 N, early stop on hash | 124 hex runs | 0.519 | 0.660 | −0.501 | 0.801 | 0.303 | 0.893 |
| **007** | **CNN** | **P3: 5 hex calib runs ×10, early stop on 5 other hex runs, 7 train indenters** | **115 hex runs** | **0.096** | **0.132** | **−0.003** | **0.992** | **0.056** | **0.9957** |
| 008 | CNN | 1 hex calib run at 3 N ×10 (+1 val run at 3 N) | 123 hex runs | 0.309 | 0.492 | −0.171 | 0.890 | 0.180 | 0.941 |
| 009 | CNN | = 007, split seed 1 (other calib runs + init) | 115 hex runs | 0.106 | 0.151 | −0.012 | 0.990 | 0.062 | 0.994 |
| 010 | CNN | = 007 recipe, test indenter heart | 115 heart runs | 0.156 | 0.205 | −0.025 | 0.981 | 0.096 | 0.990 |
| 011 | ViT | = 007 recipe | 115 hex runs | 0.300 | 0.379 | +0.279 | 0.934 | 0.175 | 0.965 |
| 012 | ViT | P2 seen indenters, unseen runs (upper bound) | runs of all 8 indenters | 0.208 | 0.302 | +0.136 | 0.953 | 0.121 | 0.975 |
| 013 | CNN | 1 hex calib run at 5 N ×10 (+1 val run at 5 N) | 123 hex runs | 0.287 | 0.361 | +0.239 | 0.939 | 0.168 | 0.968 |
| 014 | CNN | = 007, 80 epochs, patience 15 (early stop cut the LR anneal at epoch 39) | 115 hex runs | 0.127 | 0.182 | | | | |
| 015 | — | aborted at epoch 1 (same flaw as 014) | | | | | | | |
| 016 | CNN | = 007, init seed 1 (same split), full schedule | 115 hex runs | 0.105 | | | | | |
| 017 | CNN | = 007, init seed 2 (same split), full schedule | 115 hex runs | 0.095 | | | | | |
| 018 | CNN | = 007 + past frames t−3, t−6; best-val epoch (mid-schedule, epoch 20) | 115 hex runs | 0.128 | 0.180 | −0.022 | 0.985 | 0.075 | 0.992 |
| 019 | — | aborted at start (checkpoint-selection flaw, see 018) | | | | | | | |
| 020 | CNN | = 018 with the final model kept (`train.select=last`) | 115 hex runs | 0.104 | 0.146 | −0.008 | 0.990 | 0.061 | 0.995 |
| **021** | **CNN** | **= 007 + reference = mean of the first 30 frames, final model kept** | **115 hex runs** | **0.078** | **0.109** | **−0.019** | **0.995** | **0.046** | **0.997** |
| **022** | **CNN** | **= 021 recipe on heart** (cf. 010: 0.156) | **115 heart runs** | **0.074** | **0.107** | **−0.016** | **0.995** | **0.043** | **0.997** |
| 023 | ViT | = 021 recipe with the ViT (cf. 011) | 115 hex runs | 0.805 | 0.962 | +0.803 | 0.574 | 0.470 | 0.772 |

exp004 on its hexagonal test runs alone: all-frame MAE 0.056 N, R² 0.9957 (= exp007).

Training time (RTX 5070, `minutes` column of the results CSV): CNN 8–67 min (40 epochs ≈ 40 min; past-frame variants ≈ 67 min), ViT 42–143 min.

## Post-processing (no retraining; all choices made on the validation runs) – `results/ensembles/`, `results/experiments/exp021*/tta_smoothing_eval/`

| models | TTA (4 flips) | temporal smoothing | test | contact MAE | contact RMSE | contact R² | all MAE | all R² |
|---|---|---|---|---|---|---|---|---|
| 007 | no | – | hex | 0.096 | 0.132 | 0.992 | 0.056 | 0.9957 |
| 007 + 016 + 017 ensemble | yes | – | hex | 0.094 | 0.132 | 0.992 | 0.055 | 0.9957 |
| 007 | yes | median ±12 frames (offline) | hex | 0.070 | 0.095 | 0.996 | 0.041 | 0.9978 |
| 021 | yes | – | hex | 0.074 | 0.103 | 0.995 | 0.043 | 0.9974 |
| **021** | **yes** | **causal mean of 3 frames (real time)** | **hex** | **0.059** | **0.085** | **0.997** | **0.035** | **0.9982** |
| **021** | **yes** | **centred median of 25 frames (offline, 0.4 s look-ahead)** | **hex** | **0.043** | **0.057** | **0.9985** | **0.025** | **0.9992** |
| 022 | yes | causal mean of 3 frames (real time) | heart | 0.059 | 0.087 | 0.997 | 0.035 | 0.9981 |
| 022 | yes | centred median of 25 frames (offline) | heart | 0.044 | 0.059 | 0.998 | 0.026 | 0.9992 |

## Force-level classification (from the exp021 regressor, no extra training)

`results/experiments/exp021_cnn_cal5v5_hex_ref30_last/force_level_classification/`

| Task | Accuracy |
|---|---|
| Frame level, 7 classes (no contact ≤ 0.05 N, 0.05–0.5 N, 1 N bands around 1–5 N), single frame | 97.6 % (in contact 96.2 %, within one class 100 %) |
| Frame level, causal mean of 3 frames | 97.7 % (in contact 96.5 %) |
| Frame level, centred median of 25 frames | 98.6 % (in contact 97.8 %) |
| **Run level, peak force 1–5 N (115 runs)** | **100 % (115/115)**, peak-force MAE 0.055 N |

Largest frame-level confusion: 545 frames (6.7 %) of the 2 N band predicted as 1 N, where the image signal saturates.

## Indenter-shape classification (exp024)

CNN with the exp021 backbone and input, 8 classes, in-contact frames only, run-level split 70/15/15 (seed 0), 20 epochs.
`results/experiments/exp024_cnn_shape_classification/`

| Metric | Value |
|---|---|
| Frame accuracy (52,974 test frames) | **99.69 %** |
| Macro-F1 | 0.997 |
| Run accuracy (majority vote, 152 runs) | 100 % |
| Per shape | 100 % except hexagonal 99.0 %, triangle 98.7 % |
| Per force bin | 95.6 % at 0.05–0.25 N, 100 % above 0.25 N |
| Mean confidence correct / wrong | 0.955 / 0.513 |
| Validation accuracy after epoch 1 / end | 99.3 % / ≈ 99.9 % |

## Findings

1. **Strict unseen-indenter force is not identifiable from these images.** For every flat indenter the difference
   image saturates at ≈ 0.5–1 N; from 1 N to 5 N it changes only a few percent, while the saturation level differs
   strongly between shapes (hexagonal has the highest level of all). TACTO renders gel normals: once the flat face is
   fully in contact, deeper penetration (0.41 → 1.34 mm for hexagonal) barely changes the picture. Leave-one-shape-out
   regression on hand-crafted statistics is no better than a constant even with an oracle per-run normalisation, so
   neither architecture changes (exp001–003) nor temporal context can recover the absolute force of a never-seen
   indenter. All strict models are worse than a constant.
2. **With shape knowledge the images are highly informative**: seen indenters 0.083 N, R² 0.993.
3. **Few-shot calibration solves the new-indenter case**: 5 labelled runs (one per force level), oversampled ×10,
   with 5 more runs for model selection, reach the seen-indenter accuracy on the 115 untouched hexagonal runs
   (0.096 N, R² 0.992, bias −0.003 N); reproduced with another seed (0.106 N) and another indenter (heart, 0.156 N).
4. Model selection must use runs of the calibrated indenter: early stopping on hash (exp005/006) stops too early.
5. **One calibration run is not enough** for ≈ 0.1 N: at 3 N (exp008) the model is accurate up to 3 N but 0.45 N off
   at 5 N; at 5 N (exp013) it is 0.13 N at 5 N but 0.23 N at 1 N. One run per force level is needed.
6. **The CNN beats the ViT** in every setting (the ViT is trained from scratch, without pre-training).
7. **Rendering noise is the remaining error source**: every frame carries white pixel noise (σ ≈ 6.9/255); the
   predictions jittered 0.057 N per frame vs 0.010 N for the labels, and different seeds made the same errors
   (3-model ensemble 0.096 → 0.094 N only).
8. **Averaging the no-contact reference over the first 30 frames** removes the reference noise (a constant offset for
   a whole run): 0.096 → 0.078 N (exp021); the remaining per-frame noise is removed by smoothing: 0.059 N real time,
   0.043 N offline. Same on heart (exp022). At deployment: record ≈ 1 s before touching and use the mean frame as
   the reference.
9. Did not help: longer training with early stopping (exp014), past-frame input channels (exp020, 0.104 N, 1.7×
   slower), model ensembles (correlated errors). With a one-cycle schedule, best-epoch selection on the 5 small
   calibration-validation runs is noisy: keep the final model (`train.select=last`).

## Code options introduced by the experiments (defaults unchanged)

* `preprocess.diff_gain_jitter`, `model.cnn.pool` (avg | max | avgmax): strict-task attempts.
* `split.calib_runs_per_force`, `split.calib_forces`, `split.calib_val_runs_per_force`: few-shot calibration splits
  (distinct split-file names, leakage asserted).
* `train.oversample_shapes`, `train.oversample_factor`: repeat the calibration runs each epoch.
* `preprocess.ref_frames` (mean reference), `preprocess.temporal_offsets` (past-frame channels), `train.seed`
  (initialisation seed with a fixed split), `train.select` (best | last); `last_model.pt` is always saved. In
  `results.csv`, `best_epoch` is the argmin of the validation MAE even when `select=last` keeps the final epoch.
