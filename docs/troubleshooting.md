# Troubleshooting

| Problem | Cause / fix |
|---|---|
| `attempted relative import with no known parent package` | Run from the repository root through `main.py`, `run_split_experiment.py` or `experiments/*.py`, never `python tactile_force/train.py`. |
| DataLoader worker errors / hangs on Windows | `train.num_workers=0` (`run_split_experiment.py --num-workers 0`, `run_exp.py --set train.num_workers=0`). |
| CUDA out of memory | `train.batch_size=32` (or 16). |
| Training very slow on CPU | `"preprocess.size=[80,64]" train.max_frames_per_run=60` or `run_split_experiment.py --cpu-small`; H and W must be divisible by `model.vit.patch_size`. |
| `FileNotFoundError: .../split_test-hexagonal_val-hash/index/index.csv` from `run_exp.py` | Run `python run_split_experiment.py --data-root $TACTO_DATA_ROOT` first – it builds the shared index. |
| `run_exp.py` reads `/path/to/TACTO_DATASETS` | Set `TACTO_DATA_ROOT` to your dataset folder. |
| Queue scripts fail writing logs | `mkdir -p outputs/experiments/logs`. |
| `shape_classify.py` cannot find `exp021…/config_used.yaml` | It reuses the exp021 recipe: run exp021 first, or copy `checkpoints/exp021_cnn_force_final.yaml` to `outputs/experiments/exp021_cnn_cal5v5_hex_ref30_last/config_used.yaml`. |
| `label_error` in `coverage.txt` | The label CSV has no recognisable force column: set `data.force_column=<name>`. |
| Index warns that a run does not start at 0 N | The difference input needs no-contact frames at the start of every run. |
| Shapes added or removed | Delete the `split_*.json` in the index folder, re-run `index`, retrain. |
| Predictions offset by a constant on a new indenter | Expected (strict setting): calibrate with a few labelled runs of that indenter (protocol P3). |
| Noisy validation curve | Normal with the one-cycle schedule at high learning rate; keep the final model (`train.select=last`) or lower `train.lr`. |
| `pybulletX` fails with `np.float` | Already patched in `generate_tacto_dataset.py` (`np.float = float`). |
| TACTO rendering fails on a headless server | `export PYOPENGL_PLATFORM=egl` (or `osmesa`). |
| `hydra` cannot find `conf/digit.yaml` | Copy it from the TACTO examples into `data_generation/conf/` and run from `data_generation/`. |
| ESP32 not found | Set `loadcell.port` in `real_rig/config.yaml` (e.g. `COM5`, `/dev/ttyUSB0`); on Linux add your user to `dialout`. |
| No load readings | The baud rate must match the ESP32 sketch; the sketch must print `Weight: <x> g` or `W:<millis>,<grams>`. |
| DIGIT not found | Check the serial number with `digit_interface.DigitHandler.list_digits()`. |

## Reading the outputs

* `summary.json` / `compare/comparison.csv`: CNN vs ViT on the test indenter, in-contact MAE first.
* `eval_test/metrics.json`: MAE, RMSE, bias, R², 95th-percentile error – overall, in contact and without contact.
* `eval_test/metrics_by_{shape,profile,peak_force,duration}.csv`: where a model fails.
* `eval_test/parity.png`: a constant offset from the diagonal = calibration error on a new indenter; wide scatter = the
  model cannot read the indenter.
* `eval_test/runs/*.png`: measured and predicted force over time.
* `training_curves.png`: training/validation loss and validation MAE per epoch.

One held-out indenter is one sample: before claiming that one model is better, repeat with another test indenter and
2–3 seeds (as done with heart in exp010/exp022 and seeds in exp009/exp016/exp017).
