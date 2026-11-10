#!/bin/bash
# 2026-11-10 E1 check: kappa_same of terminal samples vs episode outcome, G8 checkpoints (CPU)
cd /home/junweiluo/Stoix; R=/home/junweiluo/stoix_runs/minatar_g8ck; OUT=/home/junweiluo/stoix_runs/e1_check; mkdir -p $OUT/logs
JOBS=(); for g in asterix breakout freeway seaquest space_invaders; do for s in 0 1 2; do for t in 1999360 4998400; do JOBS+=("$g $s $t"); done; done; done
NW=10
work() { local w=$1; for ((k=w; k<${#JOBS[@]}; k+=NW)); do set -- ${JOBS[$k]}
  [ -f $OUT/${1}_s${2}_t${3}.json ] && continue
  JAX_PLATFORMS=cpu taskset -c $((w*8))-$((w*8+7)) .venv/bin/python scripts_rdqn/ranking/e1_kappa_outcome.py \
    --ckpt $R/ckpt/${1}_qcw_rho_g8ck_s${2}_t${3}.pkl --jsonl $R/${1}_qcw_rho_g8ck_s${2}.jsonl --out $OUT/${1}_s${2}_t${3}.json > $OUT/logs/${1}_s${2}_t${3}.log 2>&1
done; }
for w in $(seq 0 $((NW-1))); do work $w & done; wait; date > $OUT/ALLDONE
