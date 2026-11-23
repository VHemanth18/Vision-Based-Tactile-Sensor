#!/bin/bash
# Queue 8: like queue 7 but the FINAL (annealed) model is kept (train.select=last) instead of the best epoch on
# 5 noisy calibration-val runs (exp018 picked a mid-schedule epoch). last_model.pt is now saved in every run.
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
CAL="split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 train.oversample_shapes=[hexagonal_indenter] train.early_stop_patience=999 train.select=last"
run cnn_cal5v5_hex_tmp36_last --model cnn --val none --note "= exp018 (past frames t-3,t-6), final model kept" --set $CAL "preprocess.temporal_offsets=[3,6]"
run cnn_cal5v5_hex_last --model cnn --val none --note "= exp007, final model kept" --set $CAL
run cnn_cal5v5_hex_ep80_last --model cnn --val none --note "= exp007, 80 epochs, final model kept" --set $CAL train.epochs=80
echo QUEUE8_DONE
