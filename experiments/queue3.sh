#!/bin/bash
# Experiment queue 3: calibration protocol with held-out calibration-validation runs + oversampling.
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
while pgrep -f "run_exp.py --name cnn_calib1at3N_hex" > /dev/null; do sleep 10; done   # wait for exp006
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
CAL="split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10"
run cnn_cal5v5_os10_hex       --model cnn --val none --note "train 7 shapes + 5 hex calib runs (x10), early-stop on 5 other hex runs, test 115 hex runs" --set $CAL "train.oversample_shapes=[hexagonal_indenter]"
run cnn_cal1v1at3N_os10_hex   --model cnn --val none --note "ONE hex calib run at 3 N (x10) + 1 hex val run at 3 N; test 123 hex runs" --set $CAL "split.calib_forces=[3]" "train.oversample_shapes=[hexagonal_indenter]"
run cnn_cal5v5_os10_hex_seed1 --model cnn --val none --note "as cal5v5 with split.seed=1 (different calibration runs + init)" --set $CAL split.seed=1 "train.oversample_shapes=[hexagonal_indenter]"
run cnn_cal5v5_os10_heart     --model cnn --val none --test heart_indenter --note "recipe check on another shape: test=heart" --set $CAL "train.oversample_shapes=[heart_indenter]"
run vit_cal5v5_os10_hex       --model vit --val none --note "ViT with the calibration recipe" --set $CAL "train.oversample_shapes=[hexagonal_indenter]"
run vit_runsplit_seen         --model vit --note "UPPER BOUND ViT: seen shapes, unseen runs" --set split.mode=run
echo QUEUE3_DONE
