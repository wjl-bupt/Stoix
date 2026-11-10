#!/bin/bash
# 2026-11-07 extra arm (user): CW-Q-rho on a DDPG base (one critic, no smoothing, CleanRL DDPG otherwise) =
#   --base ddpg --head qcw --sampler rho --rho 0.5 --G 8.  5 envs x 5 seeds x 1M.
# Back-fills the main queue: a worker only starts a run when fewer than MAXP runs of ours are alive (flock-serialised),
# on the GPU (of GPU_LIST) that currently hosts the fewest of our runs.  Same env vars / retries as launch_mujoco.sh v3.
cd /home/junweiluo/Stoix/scripts_mujoco
OUT=/home/junweiluo/stoix_runs/mujoco_gym; mkdir -p $OUT/_logs $OUT/_claim
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; MAXP=${MAXP:-48}; NW=${NW:-25}
PY=/home/junweiluo/Stoix/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENVS="Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5"; SEEDS="0 1 2 3 4"; a=cwq_ddpg
nproc_ours() { pgrep -fc "offpolic[y].py"; }
least_gpu() { local best= bn=999; for g in $GPU_LIST; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g 2>/dev/null | wc -l); [ $n -lt $bn ] && { bn=$n; best=$g; }; done; echo $best; }
JOBS=(); for e in $ENVS; do for s in $SEEDS; do JOBS+=("$e $s"); done; done
worker() {
  for j in "${JOBS[@]}"; do
    set -- $j; local e=$1 s=$2 f=$OUT/${1}_${a}_s${2}.jsonl
    [ -f $f ] && tail -1 $f | grep -q '"done"' && continue
    [ -d $OUT/_claim/${e}_${a}_s${s} ] && continue
    exec 9>$OUT/_claim/.lock_cwqddpg; flock 9
    while [ $(nproc_ours) -ge $MAXP ]; do sleep 20; done
    if ! mkdir $OUT/_claim/${e}_${a}_s${s} 2>/dev/null; then flock -u 9; continue; fi
    local gpu=$(least_gpu)
    ( for try in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$gpu $PY mujoco_offpolicy.py --env $e --seed $s --base ddpg --head qcw --sampler rho --rho 0.5 --G 8 --out $f >> $OUT/_logs/${e}_${a}_s${s}.log 2>&1
        rc=$?; echo "$(date +%F_%T) $e $a s$s try $try rc=$rc gpu $gpu" >> $OUT/progress.txt
        tail -1 $f | grep -q '"done"' && break
      done ) 9>&- &
    local child=$!
    sleep 20; flock -u 9            # give the new process time to appear before the next count
    wait $child
  done
}
for w in $(seq 1 $NW); do worker & sleep 1; done
wait
touch $OUT/ALLDONE_cwq_ddpg
