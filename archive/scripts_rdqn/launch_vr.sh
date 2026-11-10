#!/bin/bash
# CW-Pair MLP K=8 + unbiased variance reduction for terminal pairs (2026-11-02):
#   cwpair_mlp_strat : +system.strat_term=32                               (stratified batch, cell weights, TT cycle)
#   cwpair_mlp_multi : +system.term_partners=56                            (uniform batch, 56 extra partners per terminal item, row/col weights)
#   cwpair_mlp_sm    : +system.strat_term=32 +system.term_partners=56      (both)
# 5 games x seeds 0-4 x 5M; protocol identical to minatar_pgx cwpair_mlp (controls from /home/junweiluo/stoix_runs/minatar_pgx).
# Usage: GPU_LIST="1 3 0" NW=6 bash scripts_rdqn/launch_vr.sh   (skips DONE and running jobs; safe to start again)
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/vr; mkdir -p $OUT/_logs $OUT/_code
cp stoix/systems/q_learning/ff_rdqn.py stoix/systems/q_learning/ff_rpair.py stoix/utils/make_env.py scripts_rdqn/launch_vr.sh scripts_rdqn/check_vr.py $OUT/_code/
date >> $OUT/STARTED
GPUS=(${GPU_LIST:-1 3 0}); NW=${NW:-6}
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
Q="stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8"
arm_args() { case $1 in
  cwpair_mlp_strat) echo "+system.strat_term=32" ;;
  cwpair_mlp_multi) echo "+system.term_partners=56" ;;
  cwpair_mlp_sm)    echo "+system.strat_term=32 +system.term_partners=56" ;; esac; }
JOBS=()
for s in 0 1 2 3 4; do for g in asterix breakout freeway seaquest space_invaders; do for a in cwpair_mlp_strat cwpair_mlp_multi cwpair_mlp_sm; do JOBS+=("$g $a $s"); done; done; done
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  pgrep -f "env=pgx/$g arch.seed=$s system.algo_name=$a$" > /dev/null && return
  for attempt in $(seq 1 30); do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python $Q $(arm_args $a) env=pgx/$g arch.seed=$s system.algo_name=$a > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%F_%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%F_%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
    sleep 60
  done
}
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 3; done
wait
date > $OUT/ALLDONE
