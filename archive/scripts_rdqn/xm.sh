#!/bin/bash
cd /home/junweiluo/Stoix
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3
O="env=gymnax/asterix arch.total_timesteps=2e5 arch.num_evaluation=2 arch.seed=7 arch.absolute_metric=False"
run() { local tag=$1; shift; local t0=$(date +%s)
  timeout 400 env "$@" .venv/bin/python stoix/systems/q_learning/ff_dqn.py $O > scripts_rdqn/xm_$tag.log 2>&1
  echo "$tag rc=$? $(( $(date +%s)-t0 ))s  $(grep -o 'Episode Return[^|]*' scripts_rdqn/xm_$tag.log | tail -1)"; }
run autotune0  XLA_FLAGS="--xla_gpu_autotune_level=0"
run cmdbuf_off XLA_FLAGS="--xla_gpu_enable_command_buffer="
run both_off   XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="
echo XM_DONE
