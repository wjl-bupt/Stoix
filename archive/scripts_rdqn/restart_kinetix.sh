#!/bin/bash
for p in $(pgrep -f "launch_kineti[x].sh"); do kill -9 $p; done
for p in $(pgrep -f "run_log_dir=/home/junweiluo/stoix_runs/kinetix_of[f]"); do kill -9 $p; done
sleep 5; OUT=/home/junweiluo/stoix_runs/kinetix_off
mkdir -p $OUT/_v1_aborted; mv $OUT/small_*.jsonl $OUT/progress.txt $OUT/_logs $OUT/_v1_aborted/ 2>/dev/null; rm -rf $OUT/_claim
for f in /home/junweiluo/gpu_busy/busy_*; do p=${f##*_}; kill -0 $p 2>/dev/null || rm -f $f; done
cd /home/junweiluo/Stoix && setsid nohup bash scripts_rdqn/launch_kinetix.sh > /home/junweiluo/stoix_runs/kinetix_launch.nohup 2>&1 < /dev/null &
