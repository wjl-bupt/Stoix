#!/bin/bash
# DIAG 2026-11-01: run gap_analysis.py (--mc) on every checkpoint of every (game, seed) whose 3 algos are done.
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/diag_gap
export CUDA_VISIBLE_DEVICES=${GAP_GPU:-1} XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
ulimit -s unlimited
while true; do
  did=0
  for g in asterix seaquest; do for s in 0 1 2; do
    ok=1; for a in dqn cwpair_mlp cwpair_lr; do grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null || ok=0; done
    [ $ok = 1 ] || continue
    for f in $OUT/ckpt/${g}_dqn_s${s}_t*.pkl; do
      t=${f##*_t}; t=${t%.pkl}
      [ -f $OUT/gap/${g}_s${s}_t${t}.json ] && continue
      for k in 1 2 3 4 5 6; do
        .venv/bin/python scripts_rdqn/gap_analysis.py $OUT $g $s $t --mc > $OUT/_logs/gap_${g}_s${s}_t${t}.log 2>&1 && break
      done
      echo "$(date +%F_%T) gap $g s$s t$t $([ -f $OUT/gap/${g}_s${s}_t${t}.json ] && echo OK || echo FAIL)" >> $OUT/_logs/gap_progress.txt
      did=1
    done
  done; done
  [ -f $OUT/ALLDONE ] && [ $did = 0 ] && break
  sleep 60
done
date > $OUT/GAPDONE
