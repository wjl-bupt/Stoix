#!/bin/bash
# stop CW-Duel stage A (run as a script file; do not inline pkill -f in an ssh command line)
OUT=/home/junweiluo/stoix_runs/cwduel_A
date > $OUT/STOPPED
pkill -f "launch_cwduel_A.sh"
pkill -f "system.algo_name=cwduel_r"
echo stopped
