#!/bin/bash
# 2026-11-04 throughput profiling: one job per GPU, persistent compilation cache.
cd /home/junweiluo/Stoix; ulimit -s unlimited
export XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
O=/home/junweiluo/stoix_runs/prof_dmc; rm -rf $O; mkdir -p $O
run() { local gpu=$1 env=$2 arm=$3 e=$4
  E=$e; source scripts_rdqn/dmc_common.sh
  for t in 1 2 3 4 5 6; do
    CUDA_VISIBLE_DEVICES=$gpu timeout 3000 .venv/bin/python $(dmc_cmd $arm) env=mjc_playground/dm_control/$env \
      arch.total_timesteps=1e5 arch.num_evaluation=5 arch.seed=0 system.algo_name=${arm}_E$e system.run_log_dir=$O > $O/${env}_${arm}_E$e.log 2>&1 && break
    echo "retry $env $arm E$e $t exit" >> $O/progress.txt
  done
  echo "$env $arm E$e finished" >> $O/progress.txt; }
run 0 cheetah_run td3 16 & run 1 cheetah_run td3 64 & run 2 cartpole_balance td3 16 & run 3 cheetah_run cwq_rho 16 &
wait; date > $O/ALLDONE
