#!/bin/bash
# GPU smoke 2026-11-04: throughput + sanity for td3 / cwq_rho, 1e5 steps.  GPU0: 4 concurrent, GPU1: 1 alone.
cd /home/junweiluo/Stoix; ulimit -s unlimited
export XLA_FLAGS=--xla_gpu_autotune_level=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
O=/home/junweiluo/stoix_runs/smoke_dmc; rm -rf $O; mkdir -p $O
source scripts_rdqn/dmc_common.sh
run() { local gpu=$1 env=$2 arm=$3 tag=$4
  for t in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu timeout 3000 .venv/bin/python $(dmc_cmd $arm) env=mjc_playground/dm_control/$env \
      arch.total_timesteps=1e5 arch.num_evaluation=5 arch.seed=0 system.algo_name=${arm}_$tag system.run_log_dir=$O > $O/${env}_${arm}_$tag.log 2>&1 && break
    echo "retry $env $arm $tag $t" >> $O/progress.txt
  done
  echo "$env $arm $tag finished" >> $O/progress.txt; }
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_qcw_cont.py > $O/check_qcw_cont.out 2>&1
run 0 cheetah_run td3 g0 & run 0 cheetah_run cwq_rho g0 & run 0 hopper_hop td3 g0 & run 0 hopper_hop cwq_rho g0 &
run 1 cheetah_run cwq_rho alone &
wait; date > $O/ALLDONE
