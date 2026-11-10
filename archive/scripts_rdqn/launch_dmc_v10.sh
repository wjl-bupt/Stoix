#!/bin/bash
# 2026-11-08 v10 DMC launcher = v9 with the holder rule changed (user): no holder while our job runs on the GPU;
# the global gpu_hold_mgr.sh holds ~80% when none of our launchers marks the GPU busy (gpu_busy.sh).
# One worker per GPU in GPU_LIST; before a job: wait until >= 10 GB free (old orphaned jobs of ours / other users).
# Restart-safe: skips finished runs and runs whose python is alive; up to 30 tries.  Stop: bash scripts_rdqn/stop_dmc_v10.sh
cd /home/junweiluo/Stoix
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false" PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
OUT=${OUT:-/home/junweiluo/stoix_runs/dmc_cwq}; mkdir -p $OUT/_logs $OUT/_code $OUT/_claim10
cp stoix/systems/ddpg/cw_continuous.py scripts_rdqn/dmc_pg_common.sh scripts_rdqn/launch_dmc_v10.sh $OUT/_code/
source scripts_rdqn/dmc_pg_common.sh; source scripts_rdqn/gpu_busy.sh
GPUS=(${GPU_LIST:-0 1 2 3}); PY=.venv/bin/python
SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}; TASKS=${TASKS:-walker_walk walker_run hopper_hop finger_spin reacher_hard cheetah_run}
ARMS=${ARMS:-cwq_cmin cwq_ddpg}
JOBS=(); for s in $SEEDS; do for g in $TASKS; do for a in $ARMS; do JOBS+=("$g $a $s"); done; done; done
tname() { $PY - "$1" <<'P'
import sys, yaml; print(yaml.safe_load(open(f"stoix/configs/env/mjc_playground/dm_control/{sys.argv[1]}.yaml"))["scenario"]["task_name"])
P
}
running() { pgrep -f "dm_control/$1 .*arch.seed=$3 system.algo_name=$2 " > /dev/null; }
freemem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
worker() { local gpu=$1
  gpu_busy_on $gpu
  for j in "${JOBS[@]}"; do
    set -- $j; local g=$1 a=$2 s=$3; local tag=${g}_${a}_s$s
    local f=$OUT/$(tname $g)_${a}_s$s.jsonl
    if [ -f $f ] && tail -1 $f | grep -q '"done"'; then continue; fi
    running $g $a $s && continue
    mkdir $OUT/_claim10/$tag 2>/dev/null || continue
    for t in $(seq 1 30); do
      while [ $(freemem $gpu) -lt 10000 ]; do sleep 15; done
      local EX=""; [ $g = cheetah_run ] && EX="+env.batch_cond_reset=True +system.no_batch_vmap=True"
      echo "=== try $t $(date +%F_%T) gpu $gpu (v10)" >> $OUT/_logs/$tag.log
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        timeout 36000 $PY $(dmc_cmd $a) env=mjc_playground/dm_control/$g $EX \
        arch.total_timesteps=$(dmc_steps $g) arch.seed=$s system.algo_name=$a system.run_log_dir=$OUT >> $OUT/_logs/$tag.log 2>&1
      local rc=$?
      [ $rc != 0 ] && [ -f $f ] && tail -1 $f | grep -q '"done"' && rc=0
      echo "$(date +%F_%T) $tag try $t rc=$rc gpu $gpu" >> $OUT/progress.txt
      [ $rc = 0 ] && break; sleep $(( RANDOM % 20 ))
    done
  done
  gpu_busy_off $gpu
}
for gpu in "${GPUS[@]}"; do worker $gpu & done
wait; date > $OUT/ALLDONE_v10
