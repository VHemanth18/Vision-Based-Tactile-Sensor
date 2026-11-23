#!/bin/bash
# Queue 10: verify the final recipe (exp021: averaged 30-frame reference, final model) on a second unseen shape (heart).
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
run cnn_cal5v5_heart_ref30_last --model cnn --val none --test heart_indenter --note "= exp021 recipe on heart (cf. exp010)" --set split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 train.oversample_shapes=[heart_indenter] train.early_stop_patience=999 train.select=last preprocess.ref_frames=30
echo QUEUE10_DONE
