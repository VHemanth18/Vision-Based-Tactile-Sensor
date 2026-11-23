# Tactile force estimation: experiment log

Dataset: `/path/to/TACTO_DATASETS` (8 shapes × 125 runs, 601,000 frames, labels from the CSV
`frame` column). Env: conda `niyan2` (Python 3.11, torch 2.11.0+cu128, RTX 5070). All runs executed in tmux `niyan`.
Every experiment has its own folder `expNNN_<name>/` (config, checkpoint, history, `eval_test/`, `val_metrics.json`,
`command.txt`); `results.csv` holds one row per experiment; logs are in `logs/`. Nothing was deleted or overwritten;
exp000 (`../split_test-hexagonal_val-hash`) is read-only.

Metrics are on the TEST runs; "contact" = frames with force > 0.05 N (the README's headline metric).

| exp | model | setting | test | contact MAE (N) | contact RMSE | bias | contact R² | all-frame MAE | all R² |
|---|---|---|---|---|---|---|---|---|---|
| 000 | CNN | strict unseen shape (val hash) | hexagonal | 2.014 | 2.149 | +2.014 | −1.118 | 1.178 | −0.135 |
| 000 | ViT | strict unseen shape (val hash) | hexagonal | 2.721 | 3.132 | +2.721 | −3.500 | 1.586 | −1.412 |
| 001 | CNN | strict, diff-only input | hexagonal | 3.282 | 3.433 | +3.280 | −4.408 | 1.913 | −1.898 |
| 002 | CNN | strict, diff + gain jitter | hexagonal | 3.133 | 3.265 | +3.133 | −3.890 | 1.832 | −1.629 |
| 003 | CNN | strict, diff + jitter + avg/max pool | hexagonal | 3.804 | 4.006 | +3.804 | −6.363 | 2.223 | −2.946 |
| — | constant | predict 1.96 N (train median) on contact frames | hexagonal | 1.237 | | | | | |
| 004 | CNN | seen shapes, unseen runs (upper bound) | runs of all 8 shapes | 0.083 | 0.121 | −0.005 | 0.993 | 0.048 | 0.996 |
| 005 | CNN | 5 hex calib runs, early-stop on hash | 120 hex runs | 0.882 | 1.191 | −0.854 | 0.348 | 0.527 | 0.651 |
| 006 | CNN | 1 hex calib run @3 N, early-stop on hash | 124 hex runs | 0.519 | 0.660 | −0.501 | 0.801 | 0.303 | 0.893 |
| **007** | **CNN** | **5 hex calib runs ×10, early-stop on 5 other hex runs, 7 train shapes** | **115 hex runs** | **0.096** | **0.132** | **−0.003** | **0.992** | **0.056** | **0.9957** |
| 008 | CNN | 1 hex calib run @3 N ×10 (+1 val run @3 N) | 123 hex runs | 0.309 | 0.492 | −0.171 | 0.890 | 0.180 | 0.941 |
| 009 | CNN | = 007, seed 1 (other calib runs + init) | 115 hex runs | 0.106 | 0.151 | −0.012 | 0.990 | 0.062 | 0.994 |
| 010 | CNN | = 007 recipe, test shape heart | 115 heart runs | 0.156 | 0.205 | −0.025 | 0.981 | 0.096 | 0.990 |
| 011 | ViT | = 007 recipe | 115 hex runs | 0.300 | 0.379 | +0.279 | 0.934 | 0.175 | 0.965 |
| 012 | ViT | seen shapes, unseen runs (upper bound) | runs of all 8 shapes | 0.208 | 0.302 | +0.136 | 0.953 | 0.121 | 0.975 |
| 013 | CNN | 1 hex calib run @5 N ×10 (+1 val run @5 N) | 123 hex runs | 0.287 | 0.361 | +0.239 | 0.939 | 0.168 | 0.968 |
| 014 | CNN | = 007, 80 epochs, patience 15 (early stop cut the LR anneal at ep 39) | 115 hex runs | 0.127 | 0.182 | | | | |
| 015 | — | ABORTED at epoch 1 (same flaw as 014) | | | | | | | |
| 016 | CNN | = 007, train.seed=1 (same split), full schedule | 115 hex runs | 0.105 | | | | | |
| 017 | CNN | = 007, train.seed=2 (same split), full schedule | 115 hex runs | 0.095 | | | | | |
| 018 | CNN | = 007 + past frames t−3, t−6; best-val epoch (picked mid-schedule ep 20) | 115 hex runs | 0.128 | 0.180 | −0.022 | 0.985 | 0.075 | 0.992 |
| 019 | — | ABORTED at start (checkpoint-selection flaw, see 018) | | | | | | | |
| 020 | CNN | = 018 with the final model kept (`train.select=last`) | 115 hex runs | 0.104 | 0.146 | −0.008 | 0.990 | 0.061 | 0.995 |
| **021** | **CNN** | **= 007 + reference = mean of first 30 frames, final model kept** | **115 hex runs** | **0.078** | **0.109** | **−0.019** | **0.995** | **0.046** | **0.997** |
| **022** | **CNN** | **= 021 recipe on heart** (cf. 010: 0.156) | **115 heart runs** | **0.074** | **0.107** | **−0.016** | **0.995** | **0.043** | **0.997** |

### Post-processing (no retraining; choices made on the validation runs only) — `ensembles/*/`

| models | TTA (4 flips) | temporal smoothing | test | contact MAE | contact RMSE | contact R² | all MAE | all R² |
|---|---|---|---|---|---|---|---|---|
| 007 | no | — | hex | 0.096 | 0.132 | 0.992 | 0.056 | 0.9957 |
| 007+016+017 ensemble | yes | — | hex | 0.094 | 0.132 | 0.992 | 0.055 | 0.9957 |
| 007 | yes | median ±12 frames (offline) | hex | 0.070 | 0.095 | 0.996 | 0.041 | 0.9978 |
| 021 | yes | — | hex | 0.074 | 0.103 | 0.995 | 0.043 | 0.9974 |
| **021** | **yes** | **causal mean of 3 frames (real-time)** | **hex** | **0.059** | **0.085** | **0.997** | **0.035** | **0.9982** |
| **021** | **yes** | **centered median of 25 frames (offline, 0.4 s look-ahead)** | **hex** | **0.043** | **0.057** | **0.9985** | **0.025** | **0.9992** |
| 022 | yes | causal mean of 3 frames (real-time) | heart | 0.059 | 0.087 | 0.997 | 0.035 | 0.9981 |
| 022 | yes | centered median of 25 frames (offline) | heart | 0.044 | 0.059 | 0.998 | 0.026 | 0.9992 |

exp004 on its hexagonal test runs alone: all-frame MAE 0.056 N, R² 0.9957 (= exp007).

## Findings

1. **Strict unseen-shape force is not identifiable from these images.** For every flat indenter the image
   (frame − no-contact reference) saturates at ~0.5–1 N; from 1 N to 5 N it changes only a few percent, while the
   saturation level differs strongly between shapes and hexagonal has the highest level of all
   (`../analysis/signal_vs_force.png`). TACTO renders gel normals: once the flat face is fully in contact, deeper
   penetration (0.41 → 1.34 mm for hexagonal) barely changes the picture. Leave-one-shape-out regression on
   hand-crafted statistics (hexagonal excluded) is no better than a constant, even when each frame is normalised by
   its own run's 1 N level (an oracle) — so neither architecture changes (exp001–003) nor temporal context can
   recover absolute force for a never-seen indenter. All strict models extrapolate and are worse than a constant.
2. **With shape knowledge the images are highly informative.** Seen shapes: 0.083 N contact MAE, R² 0.993.
3. **Few-shot calibration solves the new-shape case.** 5 labelled runs of the new indenter (one per force level),
   oversampled ×10, with 5 more runs for early stopping, reach the seen-shape upper bound on the 115 untouched
   hexagonal runs (0.096 N, R² 0.992, bias −0.003 N). Reproduced with another seed (0.106 N) and another test shape
   (heart, 0.156 N).
4. Early stopping must use runs of the calibrated shape: selecting on hash (exp005/006) stops too early.
5. One calibration run is not enough for ~0.1 N: at 3 N (exp008) it is accurate up to 3 N (≤0.09 N) but 0.45 N off
   at 5 N; at 5 N (exp013) it is 0.13 N at 5 N but 0.23 N at 1 N. Both land near 0.3 N contact MAE; one run per
   force level (exp007) is what reaches ~0.1 N.
6. The CNN beats the ViT in every setting (as the README anticipates for a ViT without pretraining).
7. **Sensor noise is the remaining error source.** Every frame carries white pixel noise (σ ≈ 6.9/255, independent
   per pixel and per frame; consecutive idle frames differ by 0.031 mean-abs while contact raises the deformation
   signal only 0.031 → 0.042 and 1 N → 5 N changes it by ~0.0004). Predictions jittered 0.057 N/frame vs 0.010 N for
   the labels, and different seeds made the same errors (3-model ensemble: 0.096 → 0.094 only).
8. **Averaging the no-contact reference over the first 30 frames** (every run has ≥ 61 idle frames) removes the
   reference's noise, which is a constant offset for a whole run: 0.096 → 0.078 N single-frame (exp021), and the
   remaining per-frame noise is then removable by smoothing: 0.059 N real-time, 0.043 N offline. Same on heart
   (exp022). At deployment: record ~1 s of frames before touching and use their mean as the reference.
9. Things that did not help: 80-epoch training (needs no early stop; not pursued), past-frame input channels
   (exp020, 0.104 N, 1.7× slower), model ensembles (errors are correlated). With a OneCycle schedule, early stopping or
   best-epoch selection on the 5 small calibration-val runs is noisy: keep the final model (`train.select=last`).

## Recommended recipe (exp021)

```bash
python experiments/run_exp.py --name <name> --model cnn --val none --set \
  split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 \
  train.oversample_factor=10 "train.oversample_shapes=[hexagonal_indenter]" \
  train.early_stop_patience=999 train.select=last preprocess.ref_frames=30
python experiments/ensemble_eval.py --out outputs/experiments/ensembles/<name> --exps outputs/experiments/expNNN_<name>
```
Then use flip-TTA + causal 3-frame mean (real-time) or centered 25-frame median (offline).

## Code changes (defaults unchanged; exp000 split and checkpoint reproduce)

- `preprocess.diff_gain_jitter`, `model.cnn.pool` (avg | max | avgmax): strict-task attempts.
- `split.calib_runs_per_force`, `split.calib_forces`, `split.calib_val_runs_per_force`: few-shot calibration splits
  (distinct split-file names, leakage asserted).
- `train.oversample_shapes`, `train.oversample_factor`: repeat calibration runs each epoch (LR schedule accounts for it).
- `preprocess.ref_frames` (mean reference over the first k frames), `preprocess.temporal_offsets` (past-frame
  channels), `train.seed` (init seed with fixed split), `train.select` (best | last); `last_model.pt` always saved.
  In results.csv `best_epoch` is the argmin of val MAE even when `select=last` keeps the final epoch.
- `experiments/ensemble_eval.py` (TTA / ensembles / smoothing, selected on val), `smooth_study.py`, `queue5-10.sh`.
- `experiments/run_exp.py` (numbered runner), `feature_study.py`, `oracle_norm_study.py`, `plot_signal_vs_force.py`,
  `queue1-4.sh`.
