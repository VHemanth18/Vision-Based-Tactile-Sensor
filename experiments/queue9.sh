#!/bin/bash
# Queue 9: one targeted run. The no-contact reference was a single frame carrying pixel noise (sigma ~6.9/255) that is
# constant over the run -> average it over the first 30 frames (all runs have >= 61 idle frames before contact).
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
until grep -q '^RESULT' $L/cnn_cal5v5_hex_tmp36_last.log || grep -q Traceback $L/cnn_cal5v5_hex_tmp36_last.log; do sleep 10; done
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
CAL="split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 train.oversample_shapes=[hexagonal_indenter] train.early_stop_patience=999 train.select=last"
run cnn_cal5v5_hex_ref30_last --model cnn --val none --note "= exp007 + reference averaged over first 30 frames, final model kept" --set $CAL preprocess.ref_frames=30
echo QUEUE9_DONE
