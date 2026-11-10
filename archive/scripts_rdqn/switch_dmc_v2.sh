#!/bin/bash
# 11-05: replace launcher v1 by v2 without killing running training processes, placeholder stays off.
cd /home/junweiluo/Stoix
for p in $(pgrep -f "gpu_guard.sh bash scripts_rdqn/launch_dmc.sh"); do kill -9 $p; done   # no EXIT trap -> no placeholder
for p in $(pgrep -f "bash scripts_rdqn/launch_dmc.sh"); do kill -9 $p; done
sleep 2
echo "python left: $(pgrep -f ff_cw_td3.py | wc -l)"
cp scripts_rdqn/launch_dmc_v2.sh scripts_rdqn/launch_dmc.sh
nohup setsid bash scripts_rdqn/gpu_guard.sh bash scripts_rdqn/launch_dmc.sh > /home/junweiluo/stoix_runs/dmc_cwq_guard2.out 2>&1 < /dev/null &
echo relaunched
