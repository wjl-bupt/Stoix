#!/bin/bash
# 2026-11-07 ablations (user approved): Hopper-v5, Walker2d-v5 x seeds 0-4 x 1M, CleanRL protocol.
#   ctd_ddpg  : --base ddpg --head qct --sampler rho  (centered TD: CW-Q-rho(DDPG) without cross weighting)
#   cwq_cmin  : --base td3  --head qcw --sampler rho --min_mode centered  (CW-Q-rho(TD3) with level-free min)
# Same env vars / retries as launch_mujoco.sh v3.  GPUs 0-3, PER_GPU workers each.
cd /home/junweiluo/Stoix/scripts_mujoco
OUT=/home/junweiluo/stoix_runs/mujoco_gym; mkdir -p $OUT/_logs $OUT/_claim
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; PER_GPU=${PER_GPU:-5}
PY=/home/junweiluo/Stoix/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
argsof() { case $1 in ctd_ddpg) echo "--base ddpg --head qct --sampler rho --rho 0.5 --G 8";;
  cwq_cmin) echo "--base td3 --head qcw --sampler rho --rho 0.5 --G 8 --min_mode centered";; esac; }
JOBS=(); for e in Hopper-v5 Walker2d-v5; do for s in 0 1 2 3 4; do for a in ctd_ddpg cwq_cmin; do JOBS+=("$e $a $s"); done; done; done
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
touch $OUT/ALLDONE_ablation
