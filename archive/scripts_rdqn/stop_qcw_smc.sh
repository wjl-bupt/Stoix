#!/bin/bash
# stop the softmax-centred qcw run (launch_qcw_smc.sh + its python jobs); patterns live only in this file
L=$(pgrep -f "launch_qcw_sm[c].sh")
echo launchers: $L
[ -n "$L" ] && kill $L
sleep 2
P=$(pgrep -f "system.algo_name=qcw_rho_sm[c]$")
echo pythons: $P
[ -n "$P" ] && kill $P
sleep 5
for p in $L; do rm -fv /home/junweiluo/gpu_busy/busy_*_$p; done
date > /home/junweiluo/stoix_runs/minatar_qcw_smc/STOPPED_BY_USER_20261109
pgrep -fc "qcw_rho_sm[c]"; ls /home/junweiluo/gpu_busy/
