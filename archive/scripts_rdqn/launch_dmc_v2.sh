#!/bin/bash
# 2026-11-05 DMC main run: TD3 vs CW-Q-rho (cw_continuous.py head=q / head=qcw), protocol dmc_pg_common.sh
# (= MuJoCo Playground brax_sac_config shape, TD3 base, no obs normalisation), seeds 0-9.
# Dynamic job queue (mkdir claims), NW workers over GPU_LIST, up to 8 retries (stochastic XLA segfault at build).
# usage: nohup setsid bash scripts_rdqn/gpu_guard.sh bash scripts_rdqn/launch_dmc.sh > ... &   (restart-safe: skips done)
# XLA flags: the llvm-module / compilation parallelism + command buffer off removes the GPU compile segfault of scan(MJX step) (diag_flags.sh 11-05).
cd /home/junweiluo/Stoix
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
OUT=${OUT:-/home/junweiluo/stoix_runs/dmc_cwq}; mkdir -p $OUT/_logs $OUT/_code; rm -rf $OUT/_claim; mkdir -p $OUT/_claim
cp stoix/systems/ddpg/cw_continuous.py stoix/systems/ddpg/ff_cw_td3.py stoix/configs/system/ddpg/ff_cw_td3.yaml \
   scripts_rdqn/dmc_pg_common.sh scripts_rdqn/launch_dmc.sh scripts_rdqn/check_qcw_cont.py stoix/systems/q_learning/same_traj.py $OUT/_code/
date >> $OUT/STARTED
source scripts_rdqn/dmc_pg_common.sh
GPUS=(${GPU_LIST:-0 1 2 3}); NW=${NW:-8}
SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}; TASKS=${TASKS:-walker_walk walker_run cheetah_run hopper_hop finger_spin reacher_hard}; ARMS=${ARMS:-td3 cwq_rho}
JOBS=(); for s in $SEEDS; do for g in $TASKS; do for a in $ARMS; do JOBS+=("$g $a $s"); done; done; done
tname() { .venv/bin/python - "$1" <<'P'
import sys, yaml; print(yaml.safe_load(open(f"stoix/configs/env/mjc_playground/dm_control/{sys.argv[1]}.yaml"))["scenario"]["task_name"])
P
}
# 11-05 v2: (a) each job needs ~8.6 GB (4M replay) -> before launching, hold a per-GPU flock and wait until >= MINFREE MB
# are free, then keep the lock 150 s while the new job allocates (stops 3-per-GPU OOMs); (b) skip jobs whose python
# process is already running (survivors of v1); (c) up to 30 tries (stochastic MJX/XLA segfault at build, ~50%).
MINFREE=${MINFREE:-10000}
running() { pgrep -f "dm_control/$1 arch.total_timesteps=[0-9e.]* arch.seed=$3 system.algo_name=$2 " > /dev/null; }
gpu_wait() { while [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1) -lt $MINFREE ]; do sleep 30; done; }
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; sleep $(( w * 10 ))
  for j in "${JOBS[@]}"; do
    set -- $j; local g=$1 a=$2 s=$3; local tag=${g}_${a}_s$s
    mkdir $OUT/_claim/$tag 2>/dev/null || continue
    local f=$OUT/$(tname $g)_${a}_s$s.jsonl
    if [ -f $f ] && tail -1 $f | grep -q '"done"'; then continue; fi
    running $g $a $s && continue
    for t in $(seq 1 30); do
      exec {lk}>$OUT/_claim/gpu$gpu.lock; flock $lk; gpu_wait $gpu
      echo "=== try $t $(date +%F_%T) gpu $gpu" >> $OUT/_logs/$tag.log
      CUDA_VISIBLE_DEVICES=$gpu timeout 36000 .venv/bin/python $(dmc_cmd $a) env=mjc_playground/dm_control/$g \
        arch.total_timesteps=$(dmc_steps $g) arch.seed=$s system.algo_name=$a system.run_log_dir=$OUT >> $OUT/_logs/$tag.log 2>&1 &
      local pid=$!; sleep 150; flock -u $lk; exec {lk}>&-; wait $pid
      rc=$?; echo "$(date +%F_%T) $tag try $t rc=$rc gpu $gpu" >> $OUT/progress.txt; [ $rc = 0 ] && break; sleep $(( RANDOM % 40 ))
    done
  done; }
for w in $(seq 0 $((NW - 1))); do worker $w & done
wait; date > $OUT/ALLDONE
