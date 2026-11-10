#!/bin/bash
# asterix, seeds 0-1, CW-Pair MLP K=8 and DQN, 2M real steps, full learner state saved at the end (same protocol as minatar_pgx)
# usage: bash capture_runs.sh <gpu_cw> <gpu_dqn>
cd /home/junweiluo/Stoix; R=/home/junweiluo/stoix_runs/ledger_20261103; mkdir -p $R
P0="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e6 arch.num_evaluation=20"
P="$P0 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
run() { local a=$1 s=$2 gpu=$3 script=$4; shift 4; local D=$R/${a}_s$s; mkdir -p $D
  for att in 1 2 3 4 5 6; do
    [ -f $D/full_state.pkl ] && grep -q '"done": true' $D/*.jsonl 2>/dev/null && return
    ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python scripts_rdqn/$script $P "$@" env=pgx/asterix arch.seed=$s system.algo_name=$a system.run_log_dir=$D > $D/capture.a$att.log 2>&1 )
    echo "$(date +%F_%T) $a s$s attempt $att exit $?" >> $R/capture_progress.txt
  done; }
( run cwpair_mlp 0 $1 capture_ff_rpair.py system.head=free system.term=cross +system.n_partners=8
  run cwpair_mlp 1 $1 capture_ff_rpair.py system.head=free system.term=cross +system.n_partners=8 ) &
( run dqn 0 $2 capture_ff_rdqn.py network=minatar_dqn system.rdqn=False system.c_lr=0.0
  run dqn 1 $2 capture_ff_rdqn.py network=minatar_dqn system.rdqn=False system.c_lr=0.0 ) &
wait; date > $R/CAPTURE_DONE
