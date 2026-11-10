#!/bin/bash
# 2026-11-08: DMC (cwq_cmin+cwq_ddpg, v10) only on GPU 0,1; Kinetix on GPU 2,3 (2 each) + GPU 0,1 once DMC frees memory.
# Running python jobs are kept (both launchers skip runs whose python is alive).
for p in $(pgrep -f "launch_dmc_v1[0].sh"); do kill -9 $p; done
for p in $(pgrep -f "launch_kineti[x].sh"); do kill -9 $p; done
sleep 3
rm -rf /home/junweiluo/stoix_runs/dmc_cwq/_claim10 /home/junweiluo/stoix_runs/kinetix_off/_claim
for f in /home/junweiluo/gpu_busy/busy_*; do p=${f##*_}; kill -0 $p 2>/dev/null || rm -f $f; done
cd /home/junweiluo/Stoix
GPU_LIST="0 1" ARMS="cwq_cmin cwq_ddpg" setsid nohup bash scripts_rdqn/launch_dmc_v10.sh >> /home/junweiluo/stoix_runs/dmc_cwq/launch_v10.nohup 2>&1 < /dev/null &
setsid nohup bash scripts_rdqn/launch_kinetix.sh >> /home/junweiluo/stoix_runs/kinetix_launch.nohup 2>&1 < /dev/null &
