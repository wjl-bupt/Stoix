#!/bin/bash
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/diag_smoke; mkdir -p $OUT
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e5 arch.num_evaluation=10 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 +system.diag_grad=True +system.ckpt_every=5"
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
ulimit -s unlimited
run() { for k in $(seq 1 12); do .venv/bin/python "$@" > $OUT/log_$(echo $* | md5sum | cut -c1-6).txt 2>&1 && return; done; }
grep -q done $OUT/asterix_dqn_s0.jsonl || run stoix/systems/q_learning/ff_rdqn_diag.py $P network=minatar_dqn system.rdqn=False system.c_lr=0.0 env=pgx/asterix arch.seed=0 system.algo_name=dqn
run stoix/systems/q_learning/ff_rpair_diag.py $P system.head=free system.term=cross +system.n_partners=8 env=pgx/asterix arch.seed=0 system.algo_name=cwpair_mlp
run stoix/systems/q_learning/ff_rpair_diag.py $P system.head=lowrank system.term=cross +system.all_pairs=True +system.pair_rank=32 env=pgx/asterix arch.seed=0 system.algo_name=cwpair_lr
ls $OUT/ckpt > $OUT/ckpt_list.txt
for t in 102400 204800; do .venv/bin/python scripts_rdqn/gap_analysis.py $OUT asterix 0 $t --mc > $OUT/gap_$t.txt 2>&1; done
date > $OUT/SMOKEDONE
