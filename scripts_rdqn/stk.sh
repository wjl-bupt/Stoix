#!/bin/bash
cd /home/junweiluo/Stoix
pkill -f scripts_rdqn/smoke.sh; sleep 1
export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
O="env=gymnax/asterix arch.total_timesteps=3e5 arch.num_evaluation=3 arch.seed=7 arch.absolute_metric=False system.run_log_dir=scripts_rdqn/smoke"
for st in 8192 unlimited; do
  ( ulimit -s $st; t0=$(date +%s); timeout 500 .venv/bin/python -X faulthandler stoix/systems/q_learning/ff_rdqn.py $O system.crn=True > scripts_rdqn/stk_$st.log 2>&1; echo "stack=$st rc=$? $(( $(date +%s)-t0 ))s" )
done
echo STK_DONE
