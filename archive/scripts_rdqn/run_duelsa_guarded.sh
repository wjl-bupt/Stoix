#!/bin/bash
# 2026-11-04: runs launch_duelsa.sh on GPU0/1 and keeps the guard alive until the orphaned minatar_rho_ep8 launcher
# (old gpu_guard was SIGKILLed so that its EXIT trap does not restart the placeholder mid-run) has finished too.
# usage (detached): nohup setsid bash scripts_rdqn/gpu_guard.sh bash scripts_rdqn/run_duelsa_guarded.sh > ... &
cd /home/junweiluo/Stoix
OLD=${OLD_PID:-3412302}
GPU_LIST="0 1" NW=${NW:-8} bash scripts_rdqn/launch_duelsa.sh &
while kill -0 $OLD 2>/dev/null; do sleep 30; done
wait
