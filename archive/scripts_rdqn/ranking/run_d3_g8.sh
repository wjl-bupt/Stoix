#!/bin/bash
# 2026-11-09 D3: after minatar_g8ck/ALLDONE, run on the G=8 checkpoints (qcw_rho_g8ck s0-2, 5 games, 5 times):
#   rank_minatar.py (same args as run_ranking.sh -> ranking_g16/minatar, comparable to G16/DQN), GPU0-3, 8 at a time
#   rank_breakout_miss.py (same bank / args as run_breakout_miss.sh)
#   run_d1.sh (CPU) for the D1 kappa noise
D=/home/junweiluo/Stoix/scripts_rdqn/ranking; PY=/home/junweiluo/Stoix/.venv/bin/python
R=/home/junweiluo/stoix_runs/minatar_g8ck; O=/home/junweiluo/stoix_runs/ranking_g16; B=/home/junweiluo/stoix_runs/breakout_miss
while [ ! -f $R/ALLDONE ]; do sleep 30; done
$D/run_d1.sh > $B/d1_launcher2.log 2>&1 &
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0
cd /tmp; gi=0
for pass in 1 2; do       # pass 2 retries silent start failures (segfaults)
for t in 999680 1999360 2999040 3998720 4998400; do for g in asterix breakout freeway seaquest space_invaders; do for s in 0 1 2; do
  n=${g}_qcw_rho_g8ck_s$s; c=$R/ckpt/${n}_t$t.pkl; [ -f $c ] || continue
  out=$O/minatar/${n}_t$t.json
  if [ ! -f $out ]; then
    while [ $(pgrep -fc "rank_minata[r].py|rank_breakout_mis[s].py") -ge 8 ]; do sleep 5; done
    ( CUDA_VISIBLE_DEVICES=$((gi % 4)) $PY -I $D/rank_minatar.py --kind pair --ckpt $c --jsonl $R/$n.jsonl --out $out \
        --n_states 256 --K 32 --H 500 --chunk 16384 > $O/_logs/mn_${n}_t$t.log 2>&1 & ); gi=$((gi+1)); sleep 3
  fi
  out=$B/res/${n}_t$t.json
  if [ $g = breakout ] && [ ! -f $out ]; then
    while [ $(pgrep -fc "rank_minata[r].py|rank_breakout_mis[s].py") -ge 8 ]; do sleep 5; done
    ( CUDA_VISIBLE_DEVICES=$((gi % 4)) $PY -I $D/rank_breakout_miss.py --mode eval --ckpts $c --jsonls $R/$n.jsonl \
        --bank $B/bank.pkl --out $out --n_desc 320 --n_other 128 --K 32 --H 500 --chunk 16384 > $B/_logs/${n}_t$t.log 2>&1 & )
    gi=$((gi+1)); sleep 3
  fi
done; done; done
while pgrep -f "rank_minata[r].py|rank_breakout_mis[s].py" > /dev/null; do sleep 10; done
done
wait; touch $B/D3_DONE
