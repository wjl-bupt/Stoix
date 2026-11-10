#!/bin/bash
# 2026-11-08 Kinetix smoke + throughput: each arm 2M steps (2 evals) on GPU $1, sequential.
cd /home/junweiluo/Stoix
export XLA_FLAGS="--xla_gpu_autotune_level=0" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
source scripts_rdqn/gpu_busy.sh; G=${1:-2}; gpu_busy_on $G
OUT=/home/junweiluo/stoix_runs/kinetix_smoke; mkdir -p $OUT
KIN_NEVAL=2; source scripts_rdqn/kinetix_pg_common.sh
for a in ${ARMS:-td3 cwq_cmin sac}; do
  CUDA_VISIBLE_DEVICES=$G timeout 3600 .venv/bin/python $(kin_cmd $a) arch.total_timesteps=2e6 arch.seed=0 system.algo_name=$a \
    system.run_log_dir=$OUT > $OUT/$a.log 2>&1; echo "$a rc=$?" >> $OUT/rc.txt
done
gpu_busy_off $G; touch $OUT/ALLDONE
