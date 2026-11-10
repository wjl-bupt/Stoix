#!/bin/bash
# Equivalence check (2026-11-01): ff_rppo with rppo=False must reproduce stock ff_ppo (same seed) on asterix;
# rppo=True and rppo=True,c_lr=0 smoke.  Small budget.
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/rppo_check; rm -rf $OUT; mkdir -p $OUT
P="env=gymnax/asterix env.wrapper=null network=minatar_ppo arch.total_num_envs=128 system.rollout_length=128 system.epochs=4 system.num_minibatches=8 arch.total_timesteps=4e5 arch.num_evaluation=4 arch.absolute_metric=False arch.seed=3 logger.loggers.console.enabled=True"
export CUDA_VISIBLE_DEVICES=${GPU:-3} XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
ulimit -s unlimited
.venv/bin/python stoix/systems/ppo/anakin/ff_ppo.py $P > $OUT/stock.log 2>&1
.venv/bin/python stoix/systems/ppo/anakin/ff_rppo.py $P system.rppo=False system.run_log_dir=$OUT > $OUT/ppo.log 2>&1
.venv/bin/python stoix/systems/ppo/anakin/ff_rppo.py $P system.rppo=True system.run_log_dir=$OUT > $OUT/rppo.log 2>&1
.venv/bin/python stoix/systems/ppo/anakin/ff_rppo.py $P system.rppo=True system.c_lr=0.0 system.algo_name=rppo_c0 system.run_log_dir=$OUT > $OUT/rppo_c0.log 2>&1
date > $OUT/CHECK_DONE
