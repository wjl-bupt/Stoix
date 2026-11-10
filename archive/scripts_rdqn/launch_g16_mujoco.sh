#!/bin/bash
# 2026-11-08 (user): CW-Q-rho(DDPG) with G 16 (cwq_ddpg_g16 = --base ddpg --head qcw --sampler rho --rho .5 --G 16,
# 5 envs x s0-4) + DDPG rerun with checkpoints (ddpg_ck, s0-2).  CleanRL config, 1M steps, as mujoco_gym.
# Checkpoints (actor + critics) at 20/40/60/80/100% -> $OUT/ckpt.  All 40 runs at once on GPU0-3 (gpu_busy flags).
cd /home/junweiluo/Stoix/scripts_mujoco
OUT=/home/junweiluo/stoix_runs/mujoco_g16; mkdir -p $OUT/_logs $OUT/_code $OUT/ckpt
cp mujoco_offpolicy.py ../scripts_rdqn/launch_g16_mujoco.sh $OUT/_code/
source /home/junweiluo/Stoix/scripts_rdqn/gpu_busy.sh
GPUS=(0 1 2 3); for g in "${GPUS[@]}"; do gpu_busy_on $g; done
trap 'for g in "${GPUS[@]}"; do gpu_busy_off $g; done' EXIT
PY=/home/junweiluo/Stoix/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
JOBS=()
for e in Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5; do for s in 0 1 2 3 4; do
  JOBS+=("$e cwq_ddpg_g16 $s"); [ $s -le 2 ] && JOBS+=("$e ddpg_ck $s"); done; done
i=0
for j in "${JOBS[@]}"; do
  set -- $j; e=$1 a=$2 s=$3 f=$OUT/${1}_${2}_s${3}.jsonl; gpu=${GPUS[$((i % 4))]}; i=$((i+1))
  [ -f $f ] && tail -1 $f | grep -q '"done"' && continue
  case $a in cwq_ddpg_g16) X="--base ddpg --head qcw --sampler rho --rho 0.5 --G 16" ;; ddpg_ck) X="--base ddpg --head q --sampler uniform" ;; esac
  ( for try in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu $PY mujoco_offpolicy.py --env $e --seed $s $X --out $f --ckpt_dir $OUT/ckpt >> $OUT/_logs/${e}_${a}_s${s}.log 2>&1
      rc=$?; echo "$(date +%F_%T) $e $a s$s try $try rc=$rc gpu $gpu" >> $OUT/progress.txt
      tail -1 $f | grep -q '"done"' && break
    done ) &
  sleep 2
done
wait
touch $OUT/ALLDONE
