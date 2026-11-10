#!/bin/bash
# 2026-11-08: move to the new holder rule.  Kill old launchers + all old holders (NOT the running python jobs),
# start the holder manager, the DMC v10 launcher (cwq_cmin + cwq_ddpg on GPU 0-3) and the MuJoCo cwq_nomin launcher.
cd /home/junweiluo/Stoix
for p in $(pgrep -f "dmc_cwqddpg_star[t]"); do kill -9 $p; done
for p in $(pgrep -f "launch_dm[c].sh"); do kill -9 $p; done
pkill -f "launch_cmin[3].sh"
for p in $(pgrep -f "gpu_hold.p[y]"); do kill -9 $p; done
sleep 3
setsid nohup bash scripts_rdqn/gpu_hold_mgr.sh > /dev/null 2>&1 < /dev/null &
sleep 2
GPU_LIST="0 1 2 3" ARMS="cwq_cmin cwq_ddpg" setsid nohup bash scripts_rdqn/launch_dmc_v10.sh > /home/junweiluo/stoix_runs/dmc_cwq/launch_v10.nohup 2>&1 < /dev/null &
sleep 5
cd scripts_mujoco && setsid nohup bash launch_mujoco_nomin.sh > launch_nomin.nohup 2>&1 < /dev/null &
