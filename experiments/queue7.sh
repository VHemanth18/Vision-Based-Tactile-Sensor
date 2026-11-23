#!/bin/bash
# Queue 7: exp007 recipe trained over the FULL OneCycle schedule (no early stopping; best-val checkpoint kept).
# init-seed members (same split as exp007) for an ensemble, causal temporal input, and an 80-epoch run.
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
CAL="split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 train.oversample_shapes=[hexagonal_indenter] train.early_stop_patience=999"
run cnn_cal5v5_hex_init1 --model cnn --val none --note "= exp007, train.seed=1 (same split), full schedule" --set $CAL train.seed=1
run cnn_cal5v5_hex_init2 --model cnn --val none --note "= exp007, train.seed=2 (same split), full schedule" --set $CAL train.seed=2
run cnn_cal5v5_hex_tmp36 --model cnn --val none --note "= exp007 + past frames t-3,t-6 (causal), full schedule" --set $CAL "preprocess.temporal_offsets=[3,6]"
run cnn_cal5v5_hex_ep80_full --model cnn --val none --note "= exp007, 80 epochs, full schedule" --set $CAL train.epochs=80
echo QUEUE7_DONE
