#!/bin/bash
# 2026-11-08 (user): DMC CW-Q-rho(DDPG) (arm cwq_ddpg), 6 tasks x 10 seeds.
# GPU 0,1 now (another user's 8.1 GB process sits on both -> our holder 25% so holder+other < 15 GB and a job fits);
# GPU 2,3 join automatically when the cwq_cmin launcher there has finished (its 55% holders stay and are reused).
cd /home/junweiluo/Stoix
pkill -f "launch_cmin[3].sh"                                   # MuJoCo launcher stuck in `wait` on its own holders
H=/home/junweiluo/stoix_runs/mujoco_gym/_hold_mj
for g in 0 1; do kill -9 $(cat $H/pid_${g}_a) 2>/dev/null; rm -f $H/pid_${g}_a $H/held_${g}_a; done
sleep 3
GPU_LIST="0 1" ARMS="cwq_ddpg" HOLD_FRAC=0.25 setsid nohup bash /home/junweiluo/diag9/s5/launch_dmc.sh \
  > /home/junweiluo/stoix_runs/dmc_cwq/launch_cwqddpg01.nohup 2>&1 < /dev/null &
# wait for the cwq_cmin launcher (GPU 2,3) to finish, then add GPU 2,3 for cwq_ddpg
( while pgrep -f "launch_dm[c].sh" -a | grep -q "" && [ ! -f /home/junweiluo/stoix_runs/dmc_cwq/_cmin_done ]; do
    n=$(grep -c "cwq_cmin.*rc=0" /home/junweiluo/stoix_runs/dmc_cwq/progress.txt); [ $n -ge 60 ] && touch /home/junweiluo/stoix_runs/dmc_cwq/_cmin_done; sleep 120; done
  sleep 60
  GPU_LIST="2 3" ARMS="cwq_ddpg" setsid nohup bash /home/junweiluo/diag9/s5/launch_dmc.sh \
    > /home/junweiluo/stoix_runs/dmc_cwq/launch_cwqddpg23.nohup 2>&1 < /dev/null & ) &
