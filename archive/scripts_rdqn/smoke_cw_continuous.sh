#!/bin/bash
# CPU smoke (2026-11-04): CartpoleBalance (no termination, truncation at 1000) for ddpg/td3 x q/pair/duel (+ random pairing)
cd /home/junweiluo/Stoix; O=/home/junweiluo/stoix_runs/smoke_cw_continuous; rm -rf $O; mkdir -p $O
C="arch.total_num_envs=16 arch.total_timesteps=3.2e4 arch.num_evaluation=2 arch.num_eval_episodes=4 arch.absolute_metric=False system.warmup_steps=100 system.total_batch_size=64 system.total_buffer_size=50000 system.run_log_dir=$O"
run() { local base=$1 head=$2 pr=$3; JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=1" timeout 1800 .venv/bin/python stoix/systems/ddpg/ff_cw_$base.py env=mjc_playground/dm_control/cartpole_balance $C system.head=$head system.pairing=$pr system.algo_name=${base}_${head}_${pr} arch.seed=0 > $O/${base}_${head}_${pr}.log 2>&1; echo "$base $head $pr exit $?" >> $O/progress.txt; }
for base in ddpg td3; do
  run $base q rho &
  run $base pair rho &
  run $base duel rho &
done
run td3 pair random &
run td3 duel random &
wait; date > $O/ALLDONE
