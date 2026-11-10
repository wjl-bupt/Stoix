#!/bin/bash
# R-PPO + R-GAE arm (2026-11-01): same protocol as launch_rppo.sh, actor advantages = RVL R-GAE (zeros init).
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/minatar_rppo_10m; mkdir -p $OUT/_logs
cp stoix/systems/ppo/anakin/ff_rppo.py $OUT/ff_rppo.py.snapshot_rgae; date >> $OUT/STARTED_rgae
P="env.wrapper=null network=minatar_ppo arch.total_num_envs=128 arch.total_timesteps=1e7 arch.num_evaluation=50 arch.num_eval_episodes=128 arch.absolute_metric=False"
GPUS=(${GPUS:-2 3 4 5}); JOBS=()
for s in 0 1 2 3 4; do for g in asterix breakout; do JOBS+=("$g rppo_rgae $s"); done; done
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  for attempt in 1 2 3; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/ppo/anakin/ff_rppo.py $P env=gymnax/$g arch.seed=$s system.algo_name=$a \
        system.run_log_dir=$OUT system.rppo=True system.c_lr=1.0 system.adv_estimator=rgae > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
  done
}
NW=${NW:-10}
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 2; done
wait; date > $OUT/ALLDONE_rgae
