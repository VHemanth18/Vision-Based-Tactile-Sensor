#!/bin/bash
# Queue 11: ViT with the final recipe (exp021: averaged 30-frame reference, full schedule, final model) for a fair CNN vs ViT comparison.
cd "$(dirname "$0")/.."   # repository root
L=outputs/experiments/logs
run() { n=$1; shift; echo "=== START $n $(date +%T)"; python experiments/run_exp.py --name $n "$@" > $L/$n.log 2>&1; echo "=== END $n exit=$? $(date +%T)"; grep '^RESULT' $L/$n.log; }
run vit_cal5v5_hex_ref30_last --model vit --val none --note "= exp021 recipe with the ViT (cf. exp011)" --set split.calib_runs_per_force=1 split.calib_val_runs_per_force=1 train.oversample_factor=10 train.oversample_shapes=[hexagonal_indenter] train.early_stop_patience=999 train.select=last preprocess.ref_frames=30
echo QUEUE11_DONE
