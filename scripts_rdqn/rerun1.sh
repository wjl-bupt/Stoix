#!/bin/bash
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/minatar_base
P="env.wrapper=null network=minatar_dqn arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50"
for attempt in 4 5 6 7 8; do
  ( ulimit -s unlimited
    CUDA_VISIBLE_DEVICES=$(( attempt % 8 )) XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
    .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P env=gymnax/space_invaders arch.seed=9 system.algo_name=dqn \
      system.run_log_dir=$OUT system.rdqn=False system.c_lr=0.0 > $OUT/_logs/space_invaders_dqn_s9.a$attempt.log 2>&1 )
  if grep -q '"done": true' $OUT/space_invaders_dqn_s9.jsonl 2>/dev/null; then echo "$(date +%T) DONE space_invaders dqn s9 a$attempt" >> $OUT/_logs/progress.txt; exit; fi
  echo "$(date +%T) FAIL space_invaders dqn s9 a$attempt" >> $OUT/_logs/progress.txt
done
