#!/bin/bash
# smoke test of head=duel (CW-Duel) 2026-11-02: 0.2M steps asterix, rank 32 and 0, GPU ${GPU:-2}; records sps / memory
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/smoke_duel; mkdir -p $OUT
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e5 arch.num_evaluation=2 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 +system.diag_grad=True +system.ckpt_every=1"
export CUDA_VISIBLE_DEVICES=${GPU:-2} XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
ulimit -s unlimited
run() { for k in $(seq 1 6); do .venv/bin/python "$@" > $OUT/log_$(echo $* | md5sum | cut -c1-6).txt 2>&1 && return; echo "attempt $k failed" >> $OUT/fails.txt; done; }
( while true; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $OUT/mem.txt; sleep 15; done ) & MON=$!
run stoix/systems/q_learning/ff_rpair_diag.py $P system.head=duel system.term=cross +system.all_pairs=True +system.pair_rank=32 env=pgx/asterix arch.seed=0 system.algo_name=cwduel_r32
run stoix/systems/q_learning/ff_rpair_diag.py $P system.head=duel system.term=cross +system.all_pairs=True +system.pair_rank=0 env=pgx/asterix arch.seed=0 system.algo_name=cwduel_r0
kill $MON
date > $OUT/SMOKEDONE
