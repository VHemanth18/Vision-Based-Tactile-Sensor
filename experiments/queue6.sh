#!/bin/bash
# Experiment queue 6 (after queue 5): causal temporal input (deformation of frames t-3, t-6 as extra channels).
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
until grep -q QUEUE5_DONE $L/queue5.log; do sleep 10; done
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
CAL="split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 train.oversample_shapes=[hexagonal_indenter]"
LONG="train.epochs=80 train.early_stop_patience=15"
run cnn_cal5v5_os10_hex_ep80_tmp36 --model cnn --val none --note "= exp014 + past frames t-3,t-6 (causal)" --set $CAL $LONG "preprocess.temporal_offsets=[3,6]"
run cnn_cal5v5_os10_hex_ep80_tmp36_init1 --model cnn --val none --note "= previous, train.seed=1" --set $CAL $LONG "preprocess.temporal_offsets=[3,6]" train.seed=1
echo QUEUE6_DONE
