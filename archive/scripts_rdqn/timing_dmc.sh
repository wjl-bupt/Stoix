#!/bin/bash
# 2026-11-05 per-task throughput on the main protocol (td3 arm), 2 jobs per GPU, 1.92e5 steps (3 evals of 64k).
cd /home/junweiluo/Stoix
export XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
O=/home/junweiluo/stoix_runs/timing_dmc; rm -rf $O; mkdir -p $O
source scripts_rdqn/dmc_pg_common.sh
run() { local gpu=$1 env=$2 delay=$3; sleep $delay
  for t in 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES=$gpu timeout 3600 .venv/bin/python $(dmc_cmd td3) env=mjc_playground/dm_control/$env \
      arch.total_timesteps=1.92e5 arch.num_evaluation=3 arch.seed=0 system.algo_name=td3 system.run_log_dir=$O >> $O/$env.log 2>&1
    rc=$?; echo "$env try $t rc=$rc $(date +%H:%M:%S)" >> $O/progress.txt; [ $rc = 0 ] && break; sleep $((RANDOM % 30))
  done; }
run 0 cheetah_run 0 & run 1 hopper_hop 20 & run 2 walker_run 40 & run 3 finger_spin 60 &
run 0 cartpole_swingup 80 & run 1 ball_in_cup 100 & run 2 acrobot_swingup 120 & run 3 humanoid_walk 140 &
wait; date > $O/ALLDONE
