#!/bin/bash
# 2026-11-08 (user): CW-Q-rho(TD3) WITHOUT clipped double Q ("raw": each critic bootstraps from its own target),
# arm cwq_nomin = --base td3 --head qcw --sampler rho --min_mode none.  5 envs x 5 seeds x 1M, GPU 0-3 (shared with DMC jobs).
# Holder rule via gpu_busy.sh (no holder while running).  Stop: pkill -f launch_mujoco_nomi[n].sh; pkill -f offpolic[y].py
cd /home/junweiluo/Stoix/scripts_mujoco
OUT=/home/junweiluo/stoix_runs/mujoco_gym; mkdir -p $OUT/_logs $OUT/_claim
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; PER_GPU=${PER_GPU:-7}
PY=/home/junweiluo/Stoix/.venv/bin/python
source /home/junweiluo/Stoix/scripts_rdqn/gpu_busy.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
a=cwq_nomin
JOBS=(); for e in Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5; do for s in 0 1 2 3 4; do JOBS+=("$e $s"); done; done
worker() {
  local gpu=$1; gpu_busy_on $gpu
  for j in "${JOBS[@]}"; do
    set -- $j; local e=$1 s=$2 f=$OUT/${1}_${a}_s${2}.jsonl
    [ -f $f ] && tail -1 $f | grep -q '"done"' && continue
    mkdir $OUT/_claim/${e}_${a}_s${s} 2>/dev/null || continue
    for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu $PY mujoco_offpolicy.py --env $e --seed $s --base td3 --head qcw --sampler rho --rho 0.5 --G 8 --min_mode none --out $f >> $OUT/_logs/${e}_${a}_s${s}.log 2>&1
      rc=$?; echo "$(date +%F_%T) $e $a s$s try $try rc=$rc gpu $gpu" >> $OUT/progress.txt
      tail -1 $f | grep -q '"done"' && break
    done
  done
  gpu_busy_off $gpu
}
for g in $GPU_LIST; do for i in $(seq 1 $PER_GPU); do worker $g & sleep 2; done; done
wait
touch $OUT/ALLDONE_nomin
