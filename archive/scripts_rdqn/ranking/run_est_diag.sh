#!/bin/bash
# 2026-11-09 estimator diagnosis: fixed params + own-policy replay window, gradients under G2/G8/G16/K7 (CPU pool)
cd /home/junweiluo/Stoix; R=/home/junweiluo/stoix_runs; OUT=$R/est_diag; mkdir -p $OUT/logs
JOBS=()
for g in asterix breakout seaquest; do for src in g8ck g2; do for s in 0 1 2; do for t in 999680 1999360; do
  JOBS+=("$g $src $s $t"); done; done; done; done
NW=${NW:-12}
work() { local w=$1
  for ((k=w; k<${#JOBS[@]}; k+=NW)); do set -- ${JOBS[$k]}; local g=$1 src=$2 s=$3 t=$4
    local o=$OUT/${g}_${src}_s${s}_t${t}.json; [ -f $o ] && continue
    JAX_PLATFORMS=cpu taskset -c $((w*8))-$((w*8+7)) .venv/bin/python scripts_rdqn/ranking/est_diag.py --N ${N:-800} \
      --ckpt $R/minatar_$src/ckpt/${g}_qcw_rho_${src}_s${s}_t${t}.pkl --jsonl $R/minatar_$src/${g}_qcw_rho_${src}_s${s}.jsonl \
      --out $o > $OUT/logs/${g}_${src}_s${s}_t${t}.log 2>&1
  done; }
for w in $(seq 0 $((NW-1))); do work $w & done; wait; date > $OUT/ALLDONE
