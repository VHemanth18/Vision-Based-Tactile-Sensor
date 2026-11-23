#!/bin/bash
# Experiment queue 4 (after queue 3): minimal calibration at the top of the force range.
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
until grep -q QUEUE3_DONE $L/queue3.log; do sleep 10; done
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
CAL="split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10"
run cnn_cal1v1at5N_os10_hex --model cnn --val none --note "ONE hex calib run at 5 N (x10) + 1 hex val run at 5 N; test 123 hex runs" --set $CAL "split.calib_forces=[5]" "train.oversample_shapes=[hexagonal_indenter]"
echo QUEUE4_DONE
