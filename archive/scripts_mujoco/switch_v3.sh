#!/bin/bash
OUT=/home/junweiluo/stoix_runs/mujoco_gym
mkdir -p $OUT/_cpu_aborted && mv $OUT/*.jsonl $OUT/progress.txt $OUT/_logs $OUT/STOPPED_BY_USER_20261107 $OUT/_cpu_aborted/ 2>/dev/null; rm -rf $OUT/_claim
cd /home/junweiluo/Stoix/scripts_mujoco && setsid nohup bash launch_mujoco.sh > launch.nohup 2>&1 < /dev/null &
