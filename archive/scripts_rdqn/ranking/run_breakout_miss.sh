#!/bin/bash
# 2026-11-09 Breakout imminent-miss ranking test on minatar_g16 checkpoints (dqn_ck s0-2, qcw_rho_g16 s0-9, 5 times).
# GPU0-3 (~3.7GB free each; outside-container holders occupy the rest), 2 jobs per GPU, no preallocation.
D=/home/junweiluo/Stoix/scripts_rdqn/ranking; PY=/home/junweiluo/Stoix/.venv/bin/python
R=/home/junweiluo/stoix_runs/minatar_g16; O=/home/junweiluo/stoix_runs/breakout_miss; mkdir -p $O/res $O/_logs
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0
cd /tmp
T=4998400
if [ ! -f $O/bank.pkl ]; then
  CUDA_VISIBLE_DEVICES=0 $PY -I $D/rank_breakout_miss.py --mode bank \
    --ckpts $R/ckpt/breakout_dqn_ck_s{0,1,2}_t$T.pkl $R/ckpt/breakout_qcw_rho_g16_s{0,1,2}_t$T.pkl \
    --jsonls $R/breakout_dqn_ck_s{0,1,2}.jsonl $R/breakout_qcw_rho_g16_s{0,1,2}.jsonl \
    --out $O/bank.pkl --n_desc 600 --n_other 240 --eps 0.2 --B 64 --T 1000 > $O/_logs/bank.log 2>&1 || exit 1
fi
gi=0
for t in 4998400 3998720 2999040 1999360 999680; do
  for a in dqn_ck qcw_rho_g16; do for s in 0 1 2 3 4 5 6 7 8 9; do
    n=breakout_${a}_s$s; c=$R/ckpt/${n}_t$t.pkl; [ -f $c ] || continue
    out=$O/res/${n}_t$t.json; [ -f $out ] && continue
    while [ $(pgrep -fc "rank_breakout_mis[s].py --mode eval") -ge 8 ]; do sleep 5; done
    g=$((gi % 4)); gi=$((gi+1))
    ( CUDA_VISIBLE_DEVICES=$g $PY -I $D/rank_breakout_miss.py --mode eval --ckpts $c --jsonls $R/$n.jsonl \
        --bank $O/bank.pkl --out $out --n_desc 320 --n_other 128 --K 32 --H 500 --chunk 16384 \
        > $O/_logs/${n}_t$t.log 2>&1 & )
    sleep 3
  done; done
done
while pgrep -f "rank_breakout_mis[s].py --mode eval" > /dev/null; do sleep 10; done
touch $O/ALLDONE
