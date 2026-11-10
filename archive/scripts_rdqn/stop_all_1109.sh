#!/bin/bash
# stop AL 2x2 (launch_al_minatar.sh) and the G=2 seaquest backfill (launch_g2_minatar.sh); patterns only in this file
L=$(pgrep -f "launch_(al_minata[r]|g2_minata[r]).sh"); echo launchers: $L; [ -n "$L" ] && kill $L; sleep 2
P=$(pgrep -f "system.algo_name=(qcw_rho_a[l]|dqn_a[l]|qcw_rho_g[2])$"); echo pythons: $P; [ -n "$P" ] && kill $P; sleep 8
P2=$(pgrep -f "system.algo_name=(qcw_rho_a[l]|dqn_a[l]|qcw_rho_g[2])$"); [ -n "$P2" ] && kill -9 $P2
for p in $L; do rm -f /home/junweiluo/gpu_busy/busy_*_$p; done
date > /home/junweiluo/stoix_runs/minatar_al/STOPPED_BY_USER_20261109
date > /home/junweiluo/stoix_runs/minatar_g2/STOPPED_BY_USER_20261109
echo remaining: $(pgrep -fc "system.algo_name=(qcw_rho_a[l]|dqn_a[l]|qcw_rho_g[2])$")
ls /home/junweiluo/gpu_busy/
