#!/bin/bash
# 2026-11-08 (user): DMC CW-Q-rho(TD3) + level-free min, 10 seeds x 6 tasks, GPU 2-3 (v9 launcher, own holders stay after).
# 1) replace the MuJoCo idle holders on GPU 2/3 (90%) by the v9 launcher's own holders (55%, job shares the card).
cd /home/junweiluo/Stoix
H=/home/junweiluo/stoix_runs/mujoco_gym/_hold_mj
for g in 2 3; do kill -9 $(cat $H/pid_${g}_a) 2>/dev/null; rm -f $H/pid_${g}_a $H/held_${g}_a; done
sleep 5
GPU_LIST="2 3" ARMS="cwq_cmin" setsid nohup bash /home/junweiluo/diag9/s5/launch_dmc.sh > /home/junweiluo/stoix_runs/dmc_cwq/launch_cmin.nohup 2>&1 < /dev/null &
