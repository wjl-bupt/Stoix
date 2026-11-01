#!/bin/bash
# CW-Pair, MLP free head + ALL pairs (additive decomposition), 2026-11-01. Same protocol as launch_minatar_pgx.sh.
# 5 games x seeds 0-9 x 5M on GPU 0 only, 4 jobs at a time.  Results go into the main directory as algo cwpair_mlp_all.
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/minatar_pgx; mkdir -p $OUT/_logs $OUT/_code_mlpall
cp stoix/systems/q_learning/ff_rpair.py scripts_rdqn/launch_mlpall.sh scripts_rdqn/check_rpair.py $OUT/_code_mlpall/; date >> $OUT/STARTED_mlpall
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
GPU=0; NW=4; a=cwpair_mlp_all
JOBS=()
for s in 0 1 2 3 4 5 6 7 8 9; do for g in asterix breakout freeway seaquest space_invaders; do JOBS+=("$g $s"); done; done
run_job() {
  local g=$1 s=$2
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  for attempt in 1 2 3 4 5 6; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.all_pairs=True \
        env=pgx/$g arch.seed=$s system.algo_name=$a > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%F_%T) DONE $g $a s$s gpu$GPU a$attempt" >> $OUT/_logs/progress_mlpall.txt; return; fi
    echo "$(date +%F_%T) FAIL $g $a s$s gpu$GPU a$attempt" >> $OUT/_logs/progress_mlpall.txt
  done
}
worker() { for ((k=$1; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]}; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 2; done
wait
date > $OUT/ALLDONE_mlpall
