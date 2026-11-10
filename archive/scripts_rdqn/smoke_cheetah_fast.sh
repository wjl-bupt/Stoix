#!/bin/bash
# 2026-11-05: cheetah with batch-cond auto-reset + no update-batch vmap: throughput (td3, cwq_rho), 3 evals of 128k... 
cd /home/junweiluo/Stoix; source scripts_rdqn/dmc_pg_common.sh
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false" XLA_PYTHON_CLIENT_PREALLOCATE=false
O=/home/junweiluo/stoix_runs/smoke_cheetah_fast; mkdir -p $O
G=${G:-3}
for arm in td3 cwq_rho; do for t in 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES=$G timeout 3600 .venv/bin/python $(dmc_cmd $arm) env=mjc_playground/dm_control/cheetah_run +env.batch_cond_reset=True +system.no_batch_vmap=True \
    arch.total_timesteps=6.4e5 arch.num_evaluation=5 arch.seed=0 system.algo_name=$arm system.run_log_dir=$O >> $O/$arm.log 2>&1
  rc=$?; echo "$arm try $t rc=$rc gpu $G" >> $O/progress.txt; tail -1 $O/CheetahRun_${arm}_s0.jsonl 2>/dev/null | grep -q done && break
done; done; date > $O/ALLDONE
