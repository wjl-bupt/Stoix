#!/bin/bash
cd /home/junweiluo/Stoix
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
O="env=gymnax/asterix arch.total_timesteps=3e5 arch.num_evaluation=3 arch.seed=7 arch.absolute_metric=False"
L=scripts_rdqn/smoke; mkdir -p $L
echo "== stock ff_dqn"; date
.venv/bin/python stoix/systems/q_learning/ff_dqn.py $O > $L/stock_dqn.log 2>&1; echo rc $?
for arm in "system.rdqn=False system.crn=False system.c_lr=0.0" "system.rdqn=False system.crn=True system.c_lr=0.0" \
           "system.rdqn=True system.crn=False system.c_lr=1.0" "system.rdqn=True system.crn=True system.c_lr=1.0"; do
  echo "== ff_rdqn $arm"; date
  .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $O $arm system.run_log_dir=$L > $L/$(echo $arm | tr ' =' '__').log 2>&1; echo rc $?
done
date; echo SMOKE_DONE
