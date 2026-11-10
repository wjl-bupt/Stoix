#!/bin/bash
L=$(pgrep -f "launch_rc51_rh[o].sh"); echo launchers: $L; [ -n "$L" ] && kill $L; sleep 2
P=$(pgrep -f "system.algo_name=rc51_rh[o]$"); echo pythons: $P; [ -n "$P" ] && kill $P; sleep 6
P2=$(pgrep -f "system.algo_name=rc51_rh[o]$"); [ -n "$P2" ] && kill -9 $P2
for p in $L; do rm -f /home/junweiluo/gpu_busy/busy_*_$p; done
date > /home/junweiluo/stoix_runs/minatar_rc51_rho/PAUSED_BY_USER_20261109
echo remaining $(pgrep -fc "system.algo_name=rc51_rh[o]$"); ls /home/junweiluo/gpu_busy | grep busy
