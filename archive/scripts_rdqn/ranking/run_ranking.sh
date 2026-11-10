#!/bin/bash
# 2026-11-08 same-state ranking evals for every checkpoint of finished runs (minatar_g16: dqn_ck/qcw_rho_g16 s0-2,
# mujoco_g16: ddpg_ck/cwq_ddpg_g16 s0-2).  MuJoCo on CPU (pool PMJ); MinAtar on GPU0-3 once minatar_g16/ALLDONE exists
# (gpu_busy flags).  Re-scans every 2 min; skips existing outputs; exits when all expected outputs exist.
D=/home/junweiluo/Stoix/scripts_rdqn/ranking; PY=/home/junweiluo/Stoix/.venv/bin/python
RM=/home/junweiluo/stoix_runs/minatar_g16; RJ=/home/junweiluo/stoix_runs/mujoco_g16; O=/home/junweiluo/stoix_runs/ranking_g16
mkdir -p $O/minatar $O/mujoco $O/_logs; PMJ=${PMJ:-24}
source /home/junweiluo/Stoix/scripts_rdqn/gpu_busy.sh
done_run() { tail -1 $1 2>/dev/null | grep -q '"done"'; }
GPUS=(0 1 2 3); gi=0; held=0
while true; do
  todo=0
  for e in Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5; do for a in ddpg_ck cwq_ddpg_g16; do for s in 0 1 2; do
    for t in 200000 400000 600000 800000 1000000; do
      n=${e}_${a}_s${s}; out=$O/mujoco/${n}_t$t.json; [ -f $out ] && continue; todo=$((todo+1))
      done_run $RJ/$n.jsonl || continue; [ -f $RJ/ckpt/${n}_t$t.pkl ] || continue
      pgrep -f -- "--out $out" > /dev/null && continue
      while [ $(pgrep -fc "rank_mujoc[o].py") -ge $PMJ ]; do sleep 10; done
      ( cd /tmp; JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
        $PY $D/rank_mujoco.py --ckpt $RJ/ckpt/${n}_t$t.pkl --out $out > $O/_logs/mj_${n}_t$t.log 2>&1 & )
    done; done; done; done
  if [ -f $RM/ALLDONE ]; then
    [ $held = 0 ] && { for g in "${GPUS[@]}"; do gpu_busy_on $g; done; held=1; trap 'for g in "${GPUS[@]}"; do gpu_busy_off $g; done' EXIT; }
    for g in asterix breakout freeway seaquest space_invaders; do for a in dqn_ck qcw_rho_g16; do for s in 0 1 2; do
      for t in 999680 1999360 2999040 3998720 4998400; do
        n=${g}_${a}_s${s}; out=$O/minatar/${n}_t$t.json; [ -f $out ] && continue; todo=$((todo+1))
        done_run $RM/$n.jsonl || continue; [ -f $RM/ckpt/${n}_t$t.pkl ] || continue
        pgrep -f -- "--out $out" > /dev/null && continue
        while [ $(pgrep -fc "rank_minata[r].py") -ge 8 ]; do sleep 10; done
        k=dqn; [ $a = qcw_rho_g16 ] && k=pair; gpu=${GPUS[$((gi % 4))]}; gi=$((gi+1))
        ( CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0 \
          $PY -I $D/rank_minatar.py --kind $k --ckpt $RM/ckpt/${n}_t$t.pkl --jsonl $RM/$n.jsonl --out $out \
          --n_states 256 --K 32 --H 500 --chunk 16384 > $O/_logs/mn_${n}_t$t.log 2>&1 & )
        sleep 2
      done; done; done; done
  else todo=$((todo+1)); fi
  [ $todo = 0 ] && break
  sleep 120
done
wait; touch $O/ALLDONE
