#!/bin/bash
cd /home/junweiluo/Stoix
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_nonterm.py 2>&1 | grep -v Warning
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_rpair.py 2>&1 | tail -1
OUT=/home/junweiluo/stoix_runs/smoke_nonterm; mkdir -p $OUT
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e5 arch.num_evaluation=4 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
  timeout 600 .venv/bin/python stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 +system.nonterm_only=True env=pgx/breakout arch.seed=0 system.algo_name=smoke_nn > $OUT/log.txt 2>&1 )
tail -c 1500 $OUT/*.jsonl | grep -o '"timestep": [0-9]*\|"eval_return": [0-9.e+-]*\|"nn_pair_frac": [0-9.e+-]*\|"q_loss": [0-9.e+-]*\|"done": true' | tr '\n' ' '; echo; tail -3 $OUT/log.txt
