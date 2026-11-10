#!/bin/bash
# 2026-11-09 D1 kappa-noise diagnostic on CPU (pool 24).  CW G16 (s0-9 breakout / s0-2 others), DQN s0-2, CW G8 (minatar_g8ck, when present).
# DQN evaluated with the G of the arm it is compared to (16); its own loss has no kappa — only sd_eta is used for DQN.
D=/home/junweiluo/Stoix/scripts_rdqn/ranking; PY=/home/junweiluo/Stoix/.venv/bin/python
O=/home/junweiluo/stoix_runs/breakout_miss/d1; mkdir -p $O/_logs
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd /tmp
job() { local R=$1 n=$2 t=$3 G=$4; local c=$R/ckpt/${n}_t$t.pkl out=$O/${n}_t$t.json
  [ -f $c ] && [ ! -f $out ] || return
  while [ $(pgrep -fc "d1_kappa_nois[e].py") -ge 24 ]; do sleep 3; done
  ( $PY -I $D/d1_kappa_noise.py --ckpt $c --jsonl $R/$n.jsonl --out $out --G $G --n_batches 400 > $O/_logs/${n}_t$t.log 2>&1 & ); }
R16=/home/junweiluo/stoix_runs/minatar_g16; R8=/home/junweiluo/stoix_runs/minatar_g8ck
for t in 999680 1999360 2999040 3998720 4998400; do
  for g in asterix breakout freeway seaquest space_invaders; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      { [ $g = breakout ] || [ $s -le 2 ]; } || continue
      job $R16 ${g}_qcw_rho_g16_s$s $t 16
      [ $s -le 2 ] && job $R16 ${g}_dqn_ck_s$s $t 16
      [ $s -le 2 ] && job $R8 ${g}_qcw_rho_g8ck_s$s $t 8
    done
  done
done
while pgrep -f "d1_kappa_nois[e].py" > /dev/null; do sleep 5; done
echo done
