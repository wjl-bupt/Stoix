#!/bin/bash
cd /home/junweiluo/Stoix/scripts_mujoco
export JAX_PLATFORMS=cpu XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/junweiluo/Stoix/.venv/bin/python
mkdir -p smoke
$PY check_mujoco.py > smoke/check.out 2>&1
c=0
for env in Hopper-v5 Humanoid-v5; do
for arm in "td3 q uniform" "td3 qcw rho" "ddpg q uniform" "sac q uniform"; do
  set -- $arm
  taskset -c $((64+c)) $PY mujoco_offpolicy.py --env $env --base $1 --head $2 --sampler $3 --seed 0 --total 45000 --eval_every 5000 --eval_eps 3 --out smoke/${env}_$1_$2.jsonl > smoke/${env}_$1_$2.log 2>&1 &
  c=$((c+1))
done; done
wait
touch smoke/ALLDONE
