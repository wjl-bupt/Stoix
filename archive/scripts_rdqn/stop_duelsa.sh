#!/bin/bash
# 2026-11-04 stop head=duel_sa experiment (user).  guard killed with -9 so its trap does not restart the 2-GPU placeholder.
cd /home/junweiluo/Stoix
for p in $(pgrep -f "gpu_guard.sh bash scripts_rdqn/run_duelsa_guarded.sh") $(pgrep -f "run_duelsa_guarded.sh"); do kill -9 $p 2>/dev/null; done
pkill -9 -f "scripts_rdqn/launch_duelsa.sh"
sleep 1
pkill -f "system.algo_name=cwduelsa_rho"; sleep 5; pkill -9 -f "system.algo_name=cwduelsa_rho"
date > /home/junweiluo/stoix_runs/minatar_duelsa_ep8/STOPPED_BY_USER_20261104
