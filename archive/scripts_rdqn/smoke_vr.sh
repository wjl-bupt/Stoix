#!/bin/bash
cd /home/junweiluo/Stoix
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_vr.py 2>&1 | grep -v -i warn
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_rpair.py 2>&1 | tail -1
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_nonterm.py 2>&1 | tail -1
OUT=/home/junweiluo/stoix_runs/smoke_vr; mkdir -p $OUT
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e5 arch.num_evaluation=4 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
for arm in "strat +system.strat_term=32" "multi +system.term_partners=56" "sm +system.strat_term=32 +system.term_partners=56"; do
  set -- $arm
  ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
    timeout 900 .venv/bin/python stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 ${@:2} env=pgx/breakout arch.seed=0 system.algo_name=smoke_$1 > $OUT/log_$1.txt 2>&1 )
  echo "== $1"; grep '"timestep"' $OUT/breakout_smoke_$1_s0.jsonl | tail -1 | grep -o '"timestep": [0-9]*\|"eval_return": [0-9.e+-]*\|"pi_T": [0-9.e+-]*\|"batch_term_frac": [0-9.e+-]*\|"pw_term_cells": [0-9.e+-]*\|"sps": [0-9.]*' | tr '\n' ' '; echo; tail -1 $OUT/log_$1.txt | cut -c1-200
done
