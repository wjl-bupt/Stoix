#!/bin/bash
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/rppo_check_cpu; rm -rf $OUT; mkdir -p $OUT
P="env=gymnax/asterix env.wrapper=null network=minatar_ppo arch.total_num_envs=32 system.rollout_length=64 system.epochs=2 system.num_minibatches=4 arch.total_timesteps=4.096e4 arch.num_evaluation=2 arch.num_eval_episodes=32 arch.absolute_metric=False arch.seed=3"
export JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3
.venv/bin/python stoix/systems/ppo/anakin/ff_ppo.py $P > $OUT/stock.log 2>&1
.venv/bin/python stoix/systems/ppo/anakin/ff_ppo.py $P > $OUT/stock2.log 2>&1
.venv/bin/python stoix/systems/ppo/anakin/ff_rppo.py $P system.rppo=False system.run_log_dir=$OUT > $OUT/ppo.log 2>&1
.venv/bin/python stoix/systems/ppo/anakin/ff_rppo.py $P system.rppo=True system.run_log_dir=$OUT > $OUT/rppo.log 2>&1
date > $OUT/CHECK_DONE
