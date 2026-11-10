#!/bin/bash
# 2026-11-03 MAIN MinAtar benchmark, community update count: 5M env steps -> 1.25M gradient updates (replay ratio 1/4).
# pgx MinAtar (sticky .1), 128 env, rollout 1, epochs 32 (= 128/4 updates per vector step), batch 512, buffer 100k,
# warmup 40, Adam 5e-4, clip 10, eps 1 -> .1 over 100k steps, hard target sync every 1000 updates, eval eps 0, 50 evals.
# algos: dqn ddqn dueling c51 rdqn_rho (rho=.5) cwpair_rho (rho=.5); 5 games x seeds 0-9; seed-major order.
# Usage: GPU_LIST="0 1 ..." NW=.. bash scripts_rdqn/launch_u32.sh   (skips DONE and running jobs; safe to restart)
cd /home/junweiluo/Stoix
OUT=${OUT:-/home/junweiluo/stoix_runs/minatar_u32}; mkdir -p $OUT/_logs $OUT/_code
cp stoix/systems/q_learning/ff_rdqn.py stoix/systems/q_learning/ff_rpair.py stoix/systems/q_learning/same_traj.py \
   stoix/systems/q_learning/alpal.py stoix/configs/system/q_learning/ff_rdqn.yaml scripts_rdqn/launch_u32.sh \
   scripts_rdqn/check_rho.py scripts_rdqn/check_rho_duel.py scripts_rdqn/check_same_traj.py $OUT/_code/
git rev-parse HEAD > $OUT/_code/COMMIT 2>/dev/null; date >> $OUT/STARTED
GPUS=(${GPU_LIST:-0 1}); NW=${NW:-8}; TT=${TT:-5e6}; NE=${NE:-50}; EP=${EP:-32}
SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}; GAMES=${GAMES:-asterix breakout freeway seaquest space_invaders}
ALGOS=${ALGOS:-dqn rdqn_rho cwpair_rho ddqn c51 dueling}
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=$TT arch.num_evaluation=$NE system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=$EP system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
cmd() {
  local R="stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn"
  case $1 in
    dqn)        echo "$R system.rdqn=False system.c_lr=0.0" ;;
    ddqn)       echo "$R system.rdqn=False system.c_lr=0.0 system.double=True" ;;
    dueling)    echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dueling_dqn system.rdqn=False system.c_lr=0.0" ;;
    c51)        echo "$R system.rdqn=False system.c_lr=0.0 system.c51=True system.num_atoms=51 system.vmin=-20 system.vmax=20" ;;
    rdqn_rho)   echo "$R system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8 system.pair_rho=0.5" ;;
    cwduel_rho) echo "stoix/systems/q_learning/ff_rpair.py $P system.head=duel system.term=cross +system.all_pairs=True +system.pair_rank=32 +system.duel_cur_target=True system.same_traj=True system.st_group=8 system.pair_rho=0.5" ;;
    cwpair_rho) echo "stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.same_traj=True system.st_group=8 system.pair_rho=0.5 system.rho_kc=7" ;;
  esac
}
JOBS=()
for s in $SEEDS; do for g in $GAMES; do for a in $ALGOS; do JOBS+=("$g $a $s"); done; done; done
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
