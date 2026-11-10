#!/bin/bash
# run as a script file (the pkill pattern must not be on the ssh command line)
OUT=${OUT:-/home/junweiluo/stoix_runs/minatar_u32}
pkill -f "launch_u32.sh"; sleep 1
pkill -f 'system.algo_name=(dqn|ddqn|dueling|c51|rdqn_rho|cwpair_rho|cwduel_rho)$'
date > $OUT/STOPPED
