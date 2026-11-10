#!/bin/bash
# 2026-11-04 start head=duel_sa experiment; hand over GPU0/1 guard from the minatar_rho_ep8 run.
cd /home/junweiluo/Stoix
G=3412291; L=3412302
if tr '\0' ' ' < /proc/$G/cmdline | grep -q "gpu_guard.sh bash scripts_rdqn/launch_u32.sh"; then kill -9 $G; echo "old guard $G SIGKILLed (no EXIT trap)"; fi
kill -0 $L 2>/dev/null && echo "old launcher $L still running (will be waited for)"
OLD_PID=$L nohup setsid bash scripts_rdqn/gpu_guard.sh bash scripts_rdqn/run_duelsa_guarded.sh > /home/junweiluo/stoix_runs/minatar_duelsa_ep8_guard.out 2>&1 < /dev/null &
sleep 20; bash scripts_rdqn/placeholder.sh status
