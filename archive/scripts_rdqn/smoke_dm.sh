#!/bin/bash
# smoke of head=duel_mlp K=64 (2026-11-02): asterix + seaquest 0.2M, 4 evals, GPU ${GPU:-2}
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/smoke_dm; mkdir -p $OUT
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e5 arch.num_evaluation=4 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
export CUDA_VISIBLE_DEVICES=${GPU:-2} XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
ulimit -s unlimited
run() { for k in $(seq 1 8); do .venv/bin/python "$@" > $OUT/log_$(echo $* | md5sum | cut -c1-6).txt 2>&1 && return; echo "attempt $k failed: $*" >> $OUT/fails.txt; sleep 20; done; }
for g in asterix seaquest; do
  run stoix/systems/q_learning/ff_rpair.py $P system.head=duel_mlp system.term=cross +system.n_partners=64 env=pgx/$g arch.seed=0 system.algo_name=cwduel_mlp_k64
done
date > $OUT/SMOKEDONE
