#!/bin/bash
# CW-Duel full MinAtar run (2026-11-02 15:30): head=duel_mlp (antisymmetric MLP state-pair head + centred advantage),
# K=64 partners per transition (cyclic shifts), current-state parts of Amax/Bmin from the target network.
# 5 games x seeds 0-9 x 5M, same protocol as launch_minatar_pgx.sh (pgx MinAtar, sticky .1, 128 env, rollout 1,
# 8 epochs, batch 512, buffer 100k, Adam 5e-4, clip 10, eps 1->.1/100k, hard target sync every 1000 updates).
# usage: GPU_LIST="2 4 0 1 5 6 7" NW=7 bash launch_cwduel_full.sh      stop: bash stop_cwduel_full.sh
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/cwduel_full; mkdir -p $OUT/_logs $OUT/_code
cp stoix/systems/q_learning/ff_rpair.py scripts_rdqn/check_duel_mlp.py scripts_rdqn/launch_cwduel_full.sh $OUT/_code/
date >> $OUT/STARTED
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
GPUS=(${GPU_LIST:-2 4 0 1 5 6 7}); NW=${NW:-7}
ALGOS="cwduel_mlp_k64"
JOBS=()
for s in 0 1 2 3 4 5 6 7 8 9; do for g in asterix breakout freeway seaquest space_invaders; do for a in $ALGOS; do JOBS+=("$g $a $s"); done; done; done
cmd() {
  case $1 in
    cwduel_mlp_k64) echo "stoix/systems/q_learning/ff_rpair.py $P system.head=duel_mlp system.term=cross +system.n_partners=64" ;;
  esac
}
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  [ -f $OUT/STOPPED ] && return
  pgrep -f "env=pgx/$g arch.seed=$s system.algo_name=$a" > /dev/null && return   # already running (another launcher)
  for attempt in $(seq 1 60); do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python $(cmd $a) env=pgx/$g arch.seed=$s system.algo_name=$a > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%F_%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%F_%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
    [ -f $OUT/STOPPED ] && return
    sleep 90
  done
}
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 3; done
wait
[ -f $OUT/STOPPED ] || date > $OUT/ALLDONE
