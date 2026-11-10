#!/bin/bash
# 0.2M-step smoke of both arms on breakout + asterix (GPU $1), then prints the last records.
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/smoke_cfpair; rm -rf $OUT; mkdir -p $OUT
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e5 arch.num_evaluation=2 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
Q="stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.crn=True +system.same_coef=1.0"
for g in breakout asterix; do for cpl in crn ind; do
  for attempt in 1 2 3 4; do
    ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python $Q +system.cf_coupling=$cpl env=pgx/$g arch.seed=0 system.algo_name=cwpair_mlp_$cpl > $OUT/${g}_$cpl.log 2>&1 )
    grep -q '"done": true' $OUT/${g}_cwpair_mlp_${cpl}_s0.jsonl 2>/dev/null && break
  done
done; done
date > $OUT/SMOKE_DONE
