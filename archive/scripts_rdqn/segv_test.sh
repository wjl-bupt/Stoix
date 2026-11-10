#!/bin/bash
# 2026-11-04: does the thread-stack size matter for the cheetah compile segfault?  (glibc: RLIMIT_STACK=unlimited ->
# 2 MB pthread stacks; a finite limit -> that size).  4 jobs with ulimit -s 262144 vs 4 with unlimited, cheetah td3.
cd /home/junweiluo/Stoix
export XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
O=/home/junweiluo/stoix_runs/segv_test; rm -rf $O; mkdir -p $O
source scripts_rdqn/dmc_pg_common.sh
one() { local gpu=$1 st=$2 i=$3
  ( ulimit -s $st; CUDA_VISIBLE_DEVICES=$gpu timeout 1500 .venv/bin/python $(dmc_cmd td3) env=mjc_playground/dm_control/cheetah_run \
      arch.total_timesteps=1.28e5 arch.num_evaluation=2 arch.seed=$i system.algo_name=td3_${st}_$i system.run_log_dir=$O > $O/${st}_$i.log 2>&1
    echo "stack=$st i=$i rc=$? $(date +%H:%M:%S)" >> $O/progress.txt ) ; }
for i in 0 1 2 3; do one $i 262144 $i & one $i unlimited $((i+10)) & done
wait; date > $O/ALLDONE
