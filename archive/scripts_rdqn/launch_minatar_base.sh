#!/bin/bash
# MinAtar-protocol baselines (2026-10-31): {dqn, rdqn, ddqn, dueling} x {asterix, breakout, freeway, space_invaders}
# x seeds 0-9 x 5M env steps.  4 jobs per GPU (32 workers); a job counts as done only if its jsonl has {"done": true};
# up to 3 attempts (XLA segfaults).  Protocol: see 实验参数表 (128 envs, rollout 1, 8 updates of batch 512 per
# rollout, buffer 100k, eps 1->0.1 over 100k, hard target sync every 1000 updates, MinAtar CNN, Adam 5e-4, clip 10).
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/minatar_base; mkdir -p $OUT/_logs
git rev-parse HEAD > $OUT/COMMIT; cp stoix/systems/q_learning/ff_rdqn.py $OUT/ff_rdqn.py.snapshot; date >> $OUT/STARTED
P="env.wrapper=null network=minatar_dqn arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50"
JOBS=()
for s in 0 1 2 3 4 5 6 7 8 9; do for g in asterix breakout freeway space_invaders; do for a in dqn rdqn ddqn dueling; do JOBS+=("$g $a $s"); done; done; done
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  case $a in
    dqn)     F="system.rdqn=False system.c_lr=0.0" ;;
    rdqn)    F="system.rdqn=True system.c_lr=1.0" ;;
    ddqn)    F="system.double=True" ;;
    dueling) F="network=minatar_dueling_dqn" ;;
  esac
  for attempt in 1 2 3; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P env=gymnax/$g arch.seed=$s system.algo_name=$a \
        system.run_log_dir=$OUT $F > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
  done
}
worker() { local w=$1 gpu=$(( $1 % 8 )); for ((k=w; k<${#JOBS[@]}; k+=32)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 31); do worker $w & sleep 1; done
wait
date > $OUT/ALLDONE
