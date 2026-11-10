#!/bin/bash
# 2026-11-06 v9 DMC launcher: TD3 / CW-Q-rho / DDPG / SAC (cw_continuous.py), protocol dmc_pg_common.sh, seeds 0-9.
# One worker per GPU in GPU_LIST.  Each GPU permanently carries OUR idle holder (scripts_rdqn/gpu_hold.py, 55% = 13.5 GB,
# no kernels) next to the job (PREALLOCATE=false, ~8.6 GB) -> the card never looks free to others' grabbers.
# (11-06: while the old launcher released GPU0-3 another user took GPU0/1/3.)  No gpu_guard / user placeholder.
# Entry points import keepalive (no executable unloads, hard exit) -> the driver segfault went from ~all to 0/5.
# Restart-safe: skips finished runs and runs whose python is alive; up to 30 tries (driver segfault at build/exit;
# a segfault after the run wrote "done" counts as success).  Stop: bash scripts_rdqn/stop_dmc_v9.sh (releases holders).
cd /home/junweiluo/Stoix
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false" PYTHONFAULTHANDLER=1
export JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
OUT=${OUT:-/home/junweiluo/stoix_runs/dmc_cwq}; mkdir -p $OUT/_logs $OUT/_code $OUT/_hold; rm -rf $OUT/_claim; mkdir -p $OUT/_claim
cp stoix/systems/ddpg/cw_continuous.py stoix/systems/ddpg/ff_cw_td3.py stoix/systems/ddpg/ff_cw_sac.py stoix/systems/ddpg/ff_cw_ddpg.py \
   scripts_rdqn/dmc_pg_common.sh scripts_rdqn/launch_dmc.sh scripts_rdqn/gpu_hold.py stoix/systems/ddpg/keepalive.py $OUT/_code/
date >> $OUT/STARTED
source scripts_rdqn/dmc_pg_common.sh
GPUS=(${GPU_LIST:-2 4 5}); PY=.venv/bin/python
SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}; TASKS=${TASKS:-walker_walk walker_run hopper_hop finger_spin reacher_hard cheetah_run}
ARMS=${ARMS:-td3 cwq_rho ddpg sac}
JOBS=(); for s in $SEEDS; do for g in $TASKS; do for a in $ARMS; do JOBS+=("$g $a $s"); done; done; done
tname() { $PY - "$1" <<'P'
import sys, yaml; print(yaml.safe_load(open(f"stoix/configs/env/mjc_playground/dm_control/{sys.argv[1]}.yaml"))["scenario"]["task_name"])
P
}
running() { pgrep -f "dm_control/$1 .*arch.seed=$3 system.algo_name=$2 " > /dev/null; }
hold_start() { local g=$1 f=$OUT/_hold/held_$1; rm -f $f
  CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=${HOLD_FRAC:-0.55} \
    nohup $PY scripts_rdqn/gpu_hold.py $f >> $OUT/_hold/hold_$g.log 2>&1 < /dev/null &
  local p=$!; echo $p > $OUT/_hold/pid_$g
  for i in $(seq 1 600); do [ -f $f ] && return 0; kill -0 $p 2>/dev/null || return 1; sleep 0.2; done; return 1; }
hold_ensure() { local p=$(cat $OUT/_hold/pid_$1 2>/dev/null)
  [ -n "$p" ] && kill -0 $p 2>/dev/null && [ -f $OUT/_hold/held_$1 ] && return 0
  for i in 1 2 3 4 5 6 7 8 9 10; do hold_start $1 && return 0; sleep 1; done; echo "$(date +%F_%T) HOLD FAILED gpu $1" >> $OUT/progress.txt; }
hold_stop() { local p=$(cat $OUT/_hold/pid_$1 2>/dev/null); [ -n "$p" ] || return 0
  kill -9 $p 2>/dev/null; for i in $(seq 1 100); do kill -0 $p 2>/dev/null || break; sleep 0.1; done; rm -f $OUT/_hold/pid_$1 $OUT/_hold/held_$1; }
worker() { local gpu=$1
  hold_ensure $gpu
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu) -gt 15000 ]; do sleep 5; done   # old job of ours
  for j in "${JOBS[@]}"; do
    set -- $j; local g=$1 a=$2 s=$3; local tag=${g}_${a}_s$s
    mkdir $OUT/_claim/$tag 2>/dev/null || continue
    local f=$OUT/$(tname $g)_${a}_s$s.jsonl
    if [ -f $f ] && tail -1 $f | grep -q '"done"'; then continue; fi
    running $g $a $s && continue
    for t in $(seq 1 30); do
      local EX=""; [ $g = cheetah_run ] && EX="+env.batch_cond_reset=True +system.no_batch_vmap=True"
      echo "=== try $t $(date +%F_%T) gpu $gpu" >> $OUT/_logs/$tag.log
      hold_ensure $gpu
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        timeout 36000 $PY $(dmc_cmd $a) env=mjc_playground/dm_control/$g $EX \
        arch.total_timesteps=$(dmc_steps $g) arch.seed=$s system.algo_name=$a system.run_log_dir=$OUT >> $OUT/_logs/$tag.log 2>&1
      local rc=$?
      [ $rc != 0 ] && [ -f $f ] && tail -1 $f | grep -q '"done"' && rc=0
      echo "$(date +%F_%T) $tag try $t rc=$rc gpu $gpu" >> $OUT/progress.txt
      [ $rc = 0 ] && break; sleep $(( RANDOM % 20 ))
    done
  done
  }   # holders stay up after the queue is empty (release with stop_dmc_v9.sh)
for gpu in "${GPUS[@]}"; do worker $gpu & done
wait; date > $OUT/ALLDONE
