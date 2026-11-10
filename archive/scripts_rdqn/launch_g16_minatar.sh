#!/bin/bash
# 2026-11-08 (user): CW-Q-rho(DQN) with st_group 16 (qcw_rho_g16, 5 games x s0-9) + DQN rerun with checkpoints
# (dqn_ck, s0-2) for the same-state ranking test.  OLD protocol (epochs 8, batch 512, 128 env, 5M, 50 evals).
# Checkpoints (online params) at 20/40/60/80/100% -> $OUT/ckpt.  GPUs via gpu_busy flags.  Restart-safe.
cd /home/junweiluo/Stoix
OUT=${OUT:-/home/junweiluo/stoix_runs/minatar_g16}; mkdir -p $OUT/_logs $OUT/_code $OUT/ckpt
cp stoix/systems/q_learning/ff_rdqn.py stoix/systems/q_learning/ff_rpair.py stoix/systems/q_learning/same_traj.py scripts_rdqn/launch_g16_minatar.sh scripts_rdqn/patch_ckpt.py $OUT/_code/
date >> $OUT/STARTED
source scripts_rdqn/gpu_busy.sh
GPUS=(${GPU_LIST:-0 1 2 3}); NW=${NW:-12}; TT=${TT:-5e6}; NE=${NE:-50}; EP=${EP:-8}
GAMES=${GAMES:-asterix breakout freeway seaquest space_invaders}
for g in "${GPUS[@]}"; do gpu_busy_on $g; done
trap 'for g in "${GPUS[@]}"; do gpu_busy_off $g; done' EXIT
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=$TT arch.num_evaluation=$NE system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=$EP system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 +system.ckpt_dir=$OUT/ckpt"
cmd() {
  case $1 in
    dqn_ck)      echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=False system.c_lr=0.0" ;;
    qcw_rho_g16) echo "stoix/systems/q_learning/ff_rpair.py $P system.head=qcw system.term=cross +system.all_pairs=True +system.pair_rank=0 +system.duel_cur_target=True system.same_traj=True system.st_group=16 system.pair_rho=0.5" ;;
  esac
}
JOBS=()
for s in 0 1 2 3 4 5 6 7 8 9; do for g in $GAMES; do
  JOBS+=("$g qcw_rho_g16 $s"); [ $s -le 2 ] && JOBS+=("$g dqn_ck $s"); done; done
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  pgrep -f "env=pgx/$g arch.seed=$s system.algo_name=$a$" > /dev/null && return
  for attempt in $(seq 1 30); do
    local t0=$(date +%s)
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python $(cmd $a) env=pgx/$g arch.seed=$s system.algo_name=$a > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    local t1=$(date +%s)
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%F_%T) DONE $g $a s$s gpu$gpu a$attempt wall_sec=$((t1-t0))" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%F_%T) FAIL $g $a s$s gpu$gpu a$attempt wall_sec=$((t1-t0))" >> $OUT/_logs/progress.txt
    rm -f $OUT/${g}_${a}_s${s}.jsonl
    sleep 60
  done
}
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 3; done
wait
date > $OUT/ALLDONE
