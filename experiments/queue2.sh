#!/bin/bash
# Experiment queue 2 (starts after queue 1). Calibration variants, robustness, ViT.
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
until grep -q QUEUE1_DONE $L/queue1.log; do sleep 10; done
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
run cnn_calib1at3N_hex  --model cnn --note "few-shot: ONE hexagonal run at 3 N in train; test = other 124 hex runs" --set split.calib_runs_per_force=1 "split.calib_forces=[3]"
run cnn_calib5_hex_seed1 --model cnn --note "as calib5 but split.seed=1 (other calibration runs + init)" --set split.calib_runs_per_force=1 split.seed=1
run cnn_calib5_heart    --model cnn --test heart_indenter --note "recipe check on another shape: test=heart (7 train shapes incl. hexagonal), 5 heart calib runs" --set split.calib_runs_per_force=1
run vit_runsplit_seen   --model vit --note "UPPER BOUND ViT: seen shapes, unseen runs" --set split.mode=run
run vit_calib5_hex      --model vit --note "ViT few-shot: 5 hexagonal runs in train" --set split.calib_runs_per_force=1
echo QUEUE2_DONE
