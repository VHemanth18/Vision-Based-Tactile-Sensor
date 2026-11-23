#!/bin/bash
# Experiment queue 1 (sequential). Each run writes outputs/experiments/expNNN_<name>/ and appends results.csv
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs; mkdir -p $L
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
run cnn_diffonly        --model cnn --note "raw channels removed" --set preprocess.input_mode=diff
run cnn_diff_gainjit    --model cnn --note "diff + random diff contrast 0.6-1.6" --set preprocess.input_mode=diff "preprocess.diff_gain_jitter=[0.6,1.6]"
run cnn_diff_gainjit_avgmax --model cnn --note "diff + gain jitter + avg+max pooling" --set preprocess.input_mode=diff "preprocess.diff_gain_jitter=[0.6,1.6]" model.cnn.pool=avgmax
run cnn_runsplit_seen   --model cnn --note "UPPER BOUND: seen shapes, unseen runs (split.mode=run, all 8 shapes)" --set split.mode=run
run cnn_calib5_hex      --model cnn --note "few-shot: 5 hexagonal runs (1 per force) in train; test = other 120 hex runs" --set split.calib_runs_per_force=1
echo QUEUE1_DONE
