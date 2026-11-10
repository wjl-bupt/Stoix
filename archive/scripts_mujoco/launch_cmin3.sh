#!/bin/bash
# 2026-11-08 (user): CW-Q-rho(TD3) + level-free min on Ant/Humanoid/HalfCheetah x seeds 0-4 (15 runs) on GPU 0-1.
# GPU holders (scripts_rdqn/gpu_hold.py, idle, no kernels): GPU 2,3 at 90% memory from the start;
# GPU 0,1 at 45% from the start (jobs share the card, another user's ~8 GB process is also there),
# and after the 15 runs finish a second holder per GPU 0,1 takes another 15%.  Holders stay until stop_holders_mj.sh.
cd /home/junweiluo/Stoix/scripts_mujoco
OUT=/home/junweiluo/stoix_runs/mujoco_gym; H=$OUT/_hold_mj; mkdir -p $OUT/_logs $OUT/_claim $H
PY=/home/junweiluo/Stoix/.venv/bin/python
hold() {  # gpu fraction tag
  CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=$2 \
    setsid nohup $PY /home/junweiluo/Stoix/scripts_rdqn/gpu_hold.py $H/held_$1_$3 >> $H/hold_$1_$3.log 2>&1 < /dev/null &
  echo $! > $H/pid_$1_$3
  for i in $(seq 1 120); do [ -f $H/held_$1_$3 ] && break; sleep 1; done
  echo "$(date +%F_%T) hold gpu $1 frac $2 tag $3 ok=$([ -f $H/held_$1_$3 ] && echo 1 || echo 0)" >> $OUT/progress.txt
}
hold 2 0.90 a; hold 3 0.90 a; hold 0 0.45 a; hold 1 0.45 a
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() {  # env seed gpu
  local a=cwq_cmin f=$OUT/${1}_cwq_cmin_s${2}.jsonl
  [ -f $f ] && tail -1 $f | grep -q '"done"' && return
  mkdir $OUT/_claim/${1}_${a}_s${2} 2>/dev/null || return
  for try in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$3 $PY mujoco_offpolicy.py --env $1 --seed $2 --base td3 --head qcw --sampler rho --rho 0.5 --G 8 --min_mode centered --out $f >> $OUT/_logs/${1}_${a}_s${2}.log 2>&1
    rc=$?; echo "$(date +%F_%T) $1 $a s$2 try $try rc=$rc gpu $3" >> $OUT/progress.txt
    tail -1 $f | grep -q '"done"' && break
  done
}
i=0
for e in Humanoid-v5 Ant-v5 HalfCheetah-v5; do for s in 0 1 2 3 4; do run $e $s $((i % 2)) & i=$((i+1)); sleep 2; done; done
wait
touch $OUT/ALLDONE_cmin3
hold 0 0.15 b; hold 1 0.15 b
