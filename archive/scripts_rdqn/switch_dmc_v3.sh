#!/bin/bash
# 11-05 15:30: one job per GPU (segfault at build is ~100% when another job shares the GPU; diag_conc 3/3 vs idle 3/3 ok).
cd /home/junweiluo/Stoix
for p in $(pgrep -f "gpu_guard.sh bash scripts_rdqn/launch_dmc.sh"); do kill -9 $p; done
for p in $(pgrep -f "bash scripts_rdqn/launch_dmc.sh"); do kill -9 $p; done
sleep 2; echo "python left: $(pgrep -f ff_cw_td3.py | wc -l)"
export NW=4 MINFREE=22000
nohup setsid bash scripts_rdqn/gpu_guard.sh bash scripts_rdqn/launch_dmc.sh > /home/junweiluo/stoix_runs/dmc_cwq_guard3.out 2>&1 < /dev/null &
echo relaunched
