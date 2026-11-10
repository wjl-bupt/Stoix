#!/bin/bash
# 2026-11-06: (1) CPU smoke ddpg/sac CartpoleBalance 3.2e4 steps; (2) GPU walker_walk ddpg/sac 6.4e5 steps (learning check)
cd /home/junweiluo/Stoix; source scripts_rdqn/dmc_pg_common.sh
O=/home/junweiluo/stoix_runs/smoke_sac; rm -rf $O; mkdir -p $O
for arm in ddpg sac; do
  JAX_PLATFORMS=cpu timeout 1800 .venv/bin/python $(dmc_cmd $arm) env=mjc_playground/dm_control/cartpole_balance arch.total_num_envs=16 system.epochs=2 \
    system.total_buffer_size=100000 system.warmup_steps=32 arch.total_timesteps=3.2e4 arch.num_evaluation=2 arch.num_eval_episodes=4 \
    arch.seed=0 system.algo_name=${arm}_cpu system.run_log_dir=$O > $O/cpu_$arm.log 2>&1; echo "cpu $arm rc=$?" >> $O/progress.txt &
done; wait
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
for arm in sac ddpg; do for t in 1 2 3 4 5 6 7 8; do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | head -4 | sort -t, -k2 -nr | head -1 | cut -d, -f1)
  CUDA_VISIBLE_DEVICES=$G timeout 3600 .venv/bin/python $(dmc_cmd $arm) env=mjc_playground/dm_control/walker_walk arch.total_timesteps=6.4e5 arch.num_evaluation=5 \
    arch.seed=0 system.algo_name=$arm system.run_log_dir=$O >> $O/gpu_$arm.log 2>&1
  echo "gpu $arm try $t rc=$? gpu $G" >> $O/progress.txt; tail -1 $O/WalkerWalk_${arm}_s0.jsonl 2>/dev/null | grep -q done && break
done; done; date > $O/ALLDONE
