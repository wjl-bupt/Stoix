#!/bin/bash
cd /home/junweiluo/Stoix
export OUT=/home/junweiluo/stoix_runs/minatar_ep8_more
export INIT='GPU_LIST="0 1 2 3" NW=12 ALGOS="cwduel_rho_r0" bash scripts_rdqn/launch_ep8.sh'
mkdir -p $OUT
nohup setsid bash scripts_rdqn/gpu_guard.sh bash scripts_rdqn/run_pool.sh > $OUT/../minatar_ep8_more_guard.out 2>&1 < /dev/null &
