#!/bin/bash
# 2026-11-08 backfill for launch_g16_mujoco.sh: runs that are neither done nor running get (re)started, one at a time
# per GPU (flock), only when that GPU has >= MINFREE MB free; up to 30 tries with 60 s pauses (startup OOM retry).
cd /home/junweiluo/Stoix/scripts_mujoco
OUT=/home/junweiluo/stoix_runs/mujoco_g16; MINFREE=${MINFREE:-3000}
source /home/junweiluo/Stoix/scripts_rdqn/gpu_busy.sh
GPUS=(0 1 2 3); for g in "${GPUS[@]}"; do gpu_busy_on $g; done
trap 'for g in "${GPUS[@]}"; do gpu_busy_off $g; done' EXIT
PY=/home/junweiluo/Stoix/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
freemem() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1; }
bestgpu() { local b=0 bf=0; for g in "${GPUS[@]}"; do f=$(freemem $g); [ $f -gt $bf ] && { bf=$f; b=$g; }; done; echo "$b $bf"; }
JOBS=()
for e in Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5; do for s in 0 1 2 3 4; do
  JOBS+=("$e cwq_ddpg_g16 $s"); [ $s -le 2 ] && JOBS+=("$e ddpg_ck $s"); done; done
run() { local e=$1 a=$2 s=$3 f=$OUT/${1}_${2}_s${3}.jsonl
  case $a in cwq_ddpg_g16) X="--base ddpg --head qcw --sampler rho --rho 0.5 --G 16" ;; ddpg_ck) X="--base ddpg --head q --sampler uniform" ;; esac
  for try in $(seq 1 30); do
    exec 9>$OUT/.lock_start; flock 9
    while true; do set -- $(bestgpu); [ $2 -ge $MINFREE ] && break; sleep 30; done; gpu=$1
    ( CUDA_VISIBLE_DEVICES=$gpu $PY mujoco_offpolicy.py --env $e --seed $s $X --out $f --ckpt_dir $OUT/ckpt >> $OUT/_logs/${e}_${a}_s${s}.log 2>&1
      echo "$(date +%F_%T) $e $a s$s bf-try $try rc=$? gpu $gpu" >> $OUT/progress.txt ) 9>&- &
    local child=$!; sleep 45; flock -u 9; exec 9>&-
    wait $child
    tail -1 $f 2>/dev/null | grep -q '"done"' && return
    sleep 60
  done
}
for j in "${JOBS[@]}"; do
  set -- $j; f=$OUT/${1}_${2}_s${3}.jsonl
  [ -f $f ] && tail -1 $f | grep -q '"done"' && continue
  pgrep -f -- "--out $f " > /dev/null && continue
  run $1 $2 $3 &
  sleep 1
done
wait
touch $OUT/ALLDONE_backfill
