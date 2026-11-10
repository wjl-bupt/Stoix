#!/bin/bash
L=$(pgrep -f "launch_qcw_smcs[g].sh"); echo launchers: $L; [ -n "$L" ] && kill $L; sleep 2
P=$(pgrep -f "system.algo_name=qcw_rho_smcs[g]$"); echo pythons: $P; [ -n "$P" ] && kill $P; sleep 5
for p in $L; do rm -fv /home/junweiluo/gpu_busy/busy_*_$p; done
date > /home/junweiluo/stoix_runs/minatar_qcw_smcsg/STOPPED_BY_USER_20261109
pgrep -fc "qcw_rho_smcs[g]"; ls /home/junweiluo/gpu_busy/ | grep busy
