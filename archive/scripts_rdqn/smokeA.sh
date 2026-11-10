#!/bin/bash
cd /home/junweiluo/Stoix
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
ulimit -s unlimited
P="env.wrapper=null network=minatar_dqn arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False"
L=scripts_rdqn/smokeA; rm -rf $L; mkdir -p $L
for k in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 timeout 900 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P env=gymnax/asterix arch.seed=0 arch.total_timesteps=1e6 arch.num_evaluation=5 system.algo_name=rdqn_a system.rdqn=True system.rdqn_a=True system.c_lr=0.0 system.run_log_dir=$L > $L/a.log 2>&1
  echo "try$k rc=$?" >> $L/summary.txt; grep -q '"done": true' $L/asterix_rdqn_a_s0.jsonl 2>/dev/null && break
done
echo END >> $L/summary.txt
