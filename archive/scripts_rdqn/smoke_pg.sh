#!/bin/bash
# 2026-11-04 smoke of the Playground-SAC-aligned protocol (dmc_pg_common.sh), 3e5 steps, logs appended, faulthandler on.
cd /home/junweiluo/Stoix; ulimit -s unlimited
export XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
O=/home/junweiluo/stoix_runs/smoke_pg; rm -rf $O; mkdir -p $O
source scripts_rdqn/dmc_pg_common.sh
run() { local gpu=$1 env=$2 arm=$3
  for t in 1 2 3 4; do
    echo "=== attempt $t $(date +%H:%M:%S)" >> $O/${env}_${arm}.log
    CUDA_VISIBLE_DEVICES=$gpu timeout 3600 .venv/bin/python $(dmc_cmd $arm) env=mjc_playground/dm_control/$env \
      arch.total_timesteps=3.2e5 arch.num_evaluation=5 arch.seed=0 system.algo_name=$arm system.run_log_dir=$O >> $O/${env}_${arm}.log 2>&1
    rc=$?; echo "$env $arm attempt $t rc=$rc $(date +%H:%M:%S)" >> $O/progress.txt; [ $rc = 0 ] && break
  done; }
run 0 cheetah_run td3 & run 1 cheetah_run cwq_rho & run 2 walker_walk cwq_rho & run 3 reacher_hard cwq_rho &
wait; date > $O/ALLDONE
