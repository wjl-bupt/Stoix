#!/bin/bash
pkill -f "launch_mujoc[o].sh"; sleep 1; pkill -f "offpolic[y].py"; sleep 3
OUT=/home/junweiluo/stoix_runs/mujoco_gym
mkdir -p $OUT/_v1_aborted && mv $OUT/*.jsonl $OUT/progress.txt $OUT/_v1_aborted/ 2>/dev/null; rm -rf $OUT/_claim
cd /home/junweiluo/Stoix/scripts_mujoco && setsid nohup bash launch_mujoco.sh > launch.nohup 2>&1 < /dev/null &
