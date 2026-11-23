#!/bin/bash
# Experiment queue 5: improve on exp007. Same split as exp007 (split seed 42); longer training; init seeds for an ensemble.
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
CAL="split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 train.oversample_shapes=[hexagonal_indenter]"
LONG="train.epochs=80 train.early_stop_patience=15"
run cnn_cal5v5_os10_hex_ep80 --model cnn --val none --note "= exp007, 80 epochs" --set $CAL $LONG
run cnn_cal5v5_os10_hex_ep80_init1 --model cnn --val none --note "= exp014, train.seed=1 (same split)" --set $CAL $LONG train.seed=1
run cnn_cal5v5_os10_hex_ep80_init2 --model cnn --val none --note "= exp014, train.seed=2 (same split)" --set $CAL $LONG train.seed=2
echo QUEUE5_DONE
