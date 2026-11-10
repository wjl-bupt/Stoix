#!/bin/bash
# Reset-continuing (2026-10-31): termination -> transition to the reset state, done stored as 0.
# {dqn_reset, rdqn_reset} x 4 MinAtar x seeds 0-4 x 5M; protocol identical to minatar_base (controls reused).
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/minatar_reset; mkdir -p $OUT/_logs
cp stoix/systems/q_learning/ff_rdqn.py $OUT/ff_rdqn.py.snapshot; cp scripts_rdqn/check_reset.py $OUT/; date >> $OUT/STARTED
P="env.wrapper=null network=minatar_dqn arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50"
GPUS=(4 5 6 7); NW=16
JOBS=()
for s in 0 1 2 3 4; do for g in asterix breakout freeway space_invaders; do for a in dqn_reset rdqn_reset; do JOBS+=("$g $a $s"); done; done; done
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  case $a in
    dqn_reset)  F="system.rdqn=False system.c_lr=0.0 system.reset_continuing=True" ;;
    rdqn_reset) F="system.rdqn=True system.c_lr=1.0 system.reset_continuing=True" ;;
  esac
  for attempt in 1 2 3; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P env=gymnax/$g arch.seed=$s system.algo_name=$a \
        system.run_log_dir=$OUT $F > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
  done
}
worker() { local w=$1 gpu=${GPUS[$(( $1 % 4 ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 1; done
wait
date > $OUT/ALLDONE
