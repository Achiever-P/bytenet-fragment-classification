#!/bin/bash
# usage: run_job.sh name seed extra-args...
export OMP_NUM_THREADS=1
n=$1; s=$2; shift 2
python3 -m bytenet.train --name $n --seed $s --out results/$n "$@" > logs/$n.log 2>&1
tail -1 logs/$n.log
