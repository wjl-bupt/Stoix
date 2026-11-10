#!/bin/bash
# 2026-11-07 v3 (GPU)  Gym MuJoCo v5, CleanRL-default protocol: 4 arms x 5 envs x 5 seeds x 1M steps.
# Dynamic queue, PER_GPU workers on each GPU in GPU_LIST (default GPUs 0-3, 12 per GPU = 48 concurrent runs).
# Measured (Hopper TD3, 40k steps): 1 run/GPU 285 sps, 8 runs/GPU 242 sps each, 16 runs/GPU 100-170 sps each.
# Driver workaround: KEEPALIVE=1 (never unload executables, os._exit at the end), autotune off.
# Slowest runs first.  Retries 3x if no "done".  Skips runs already "done".  Stop: bash stop_mujoco.sh
cd /home/junweiluo/Stoix/scripts_mujoco
OUT=/home/junweiluo/stoix_runs/mujoco_gym; mkdir -p $OUT/_logs $OUT/_claim
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; PER_GPU=${PER_GPU:-12}
PY=/home/junweiluo/Stoix/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENVS="Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5"
ARMS="sac cwq_rho td3 ddpg"
SEEDS="0 1 2 3 4"
argsof() { case $1 in td3) echo "--base td3 --head q --sampler uniform";; cwq_rho) echo "--base td3 --head qcw --sampler rho --rho 0.5 --G 8";;
  ddpg) echo "--base ddpg --head q --sampler uniform";; sac) echo "--base sac --head q --sampler uniform";; esac; }
JOBS=(); for e in $ENVS; do for s in $SEEDS; do for a in $ARMS; do JOBS+=("$e $a $s"); done; done; done
worker() {
  local gpu=$1
  for j in "${JOBS[@]}"; do
    set -- $j; local e=$1 a=$2 s=$3 f=$OUT/${1}_${2}_s${3}.jsonl
    [ -f $f ] && tail -1 $f | grep -q '"done"' && continue
    mkdir $OUT/_claim/${e}_${a}_s${s} 2>/dev/null || continue
    for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu $PY mujoco_offpolicy.py --env $e --seed $s $(argsof $a) --out $f >> $OUT/_logs/${e}_${a}_s${s}.log 2>&1
      rc=$?; echo "$(date +%F_%T) $e $a s$s try $try rc=$rc gpu $gpu" >> $OUT/progress.txt
      tail -1 $f | grep -q '"done"' && break
    done
  done
}
for g in $GPU_LIST; do for i in $(seq 1 $PER_GPU); do worker $g & sleep 2; done; done
wait
touch $OUT/ALLDONE
