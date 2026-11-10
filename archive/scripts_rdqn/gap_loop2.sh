#!/bin/bash
# DIAG 2026-11-01 v2: gap_analysis2.py --mc for every checkpoint step t once all 9 agents (3 algos x seeds 0-2) of a game are done.
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/diag_gap
export CUDA_VISIBLE_DEVICES=${GAP_GPU:-1} XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
ulimit -s unlimited
while true; do
  did=0
  for g in asterix seaquest; do
    ok=1; for s in 0 1 2; do for a in dqn cwpair_mlp cwpair_lr; do grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null || ok=0; done; done
    [ $ok = 1 ] || continue
    for f in $OUT/ckpt/${g}_dqn_s0_t*.pkl; do
      t=${f##*_t}; t=${t%.pkl}
      [ -f $OUT/gap2/${g}_t${t}.json ] && continue
      for k in $(seq 1 25); do
        .venv/bin/python scripts_rdqn/gap_analysis2.py $OUT $g $t --mc > $OUT/_logs/gap2_${g}_t${t}.log 2>&1; [ -f $OUT/gap2/${g}_t${t}.json ] && break
      done
      echo "$(date +%F_%T) gap2 $g t$t $([ -f $OUT/gap2/${g}_t${t}.json ] && echo OK || echo FAIL)" >> $OUT/_logs/gap_progress.txt
      did=1
    done
  done
  [ -f $OUT/ALLDONE ] && [ $did = 0 ] && break
  sleep 60
done
date > $OUT/GAPDONE
