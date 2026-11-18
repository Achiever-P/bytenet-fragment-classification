#!/bin/bash
# Trains all 8 configurations x 3 seeds (24 runs), two at a time, then aggregates.
mkdir -p logs results
jobs=()
for s in 1 2 3; do
  jobs+=("base_s$s $s --variant resnet --side byte --fusion concat")
  jobs+=("imgonly_s$s $s --ablation image_only")
  jobs+=("byteonly_s$s $s --side byte --ablation side_only")
  jobs+=("histonly_s$s $s --side hist --ablation side_only")
  jobs+=("hist_s$s $s --side hist --fusion concat")
  jobs+=("histgate_s$s $s --side hist --fusion gate")
  jobs+=("former_base_s$s $s --variant poolformer --side byte --epochs 60")
  jobs+=("former_histgate_s$s $s --variant poolformer --side hist --fusion gate --epochs 60")
done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -L 1 ./run_job.sh
python3 aggregate.py
