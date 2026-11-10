#!/bin/bash
# 2026-11-08 Kinetix off-policy main run (kinetix_pg_common.sh protocol): arms td3 cwq_cmin ddpg cwq_ddpg sac x seeds 0-2,
# 1e8 env steps.  Dynamic queue, PER_GPU workers on each GPU of GPU_LIST, a job starts only if >= MINFREE MB free.
# Holder rule via gpu_busy.sh.  Skips finished runs.  Stop: pkill -f launch_kineti[x].sh; pkill -f "run_log_dir=/home/junweiluo/stoix_runs/kinetix_of[f]"
cd /home/junweiluo/Stoix
# v2 11-08: same XLA flags as the DMC launcher (driver segfault workarounds) + per-GPU start lock (one compile at a time,
# hold 150 s so memory is allocated before the next free-memory check) + 30 tries (rc 139 = driver segfault, retry).
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
source scripts_rdqn/gpu_busy.sh; source scripts_rdqn/kinetix_pg_common.sh
OUT=/home/junweiluo/stoix_runs/kinetix_off; mkdir -p $OUT/_logs $OUT/_claim
cp scripts_rdqn/kinetix_pg_common.sh scripts_rdqn/launch_kinetix.sh stoix/systems/ddpg/cw_continuous.py stoix/networks/specialised/kinetix_tf.py $OUT/
# v3 11-08: workers per GPU given by WORKERS (gpu list with repeats); a Kinetix run takes ~10-12 GB -> MINFREE 11000.
WORKERS=${WORKERS:-"2 2 3 3 0 1"}; MINFREE=${MINFREE:-11000}
ARMS=${ARMS:-"cwq_cmin td3 sac cwq_ddpg ddpg"}; SEEDS=${SEEDS:-"0 1 2"}
JOBS=(); for s in $SEEDS; do for a in $ARMS; do JOBS+=("$a $s"); done; done
freemem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
worker() { local gpu=$1
  gpu_busy_on $gpu
  for j in "${JOBS[@]}"; do
    set -- $j; local a=$1 s=$2 f=$OUT/small_${1}_s${2}.jsonl
    [ -f $f ] && tail -1 $f | grep -q '"done"' && continue
    pgrep -f "algo_name=$a system.run_log_dir=$OUT" -a | grep -q "arch.seed=$s " && continue   # still running (old launcher)
    mkdir $OUT/_claim/${a}_s$s 2>/dev/null || continue
    for t in $(seq 1 30); do
      exec 8>$OUT/_claim/.lock_gpu$gpu; flock 8
      while [ $(freemem $gpu) -lt $MINFREE ]; do sleep 20; done
      ( CUDA_VISIBLE_DEVICES=$gpu timeout 86400 .venv/bin/python $(kin_cmd $a) arch.total_timesteps=1e8 arch.seed=$s \
        system.algo_name=$a system.run_log_dir=$OUT >> $OUT/_logs/${a}_s$s.log 2>&1; echo $? > $OUT/_claim/rc_${a}_s$s ) 8>&- &
      local child=$!; sleep 150; flock -u 8; exec 8>&-
      wait $child; rc=$(cat $OUT/_claim/rc_${a}_s$s)
      [ $rc != 0 ] && tail -1 $f | grep -q '"done"' && rc=0
      echo "$(date +%F_%T) $a s$s try $t rc=$rc gpu $gpu" >> $OUT/progress.txt
      [ $rc = 0 ] && break; sleep $(( RANDOM % 30 ))
    done
  done
  gpu_busy_off $gpu
}
for g in $WORKERS; do worker $g & sleep 2; done
wait; touch $OUT/ALLDONE
