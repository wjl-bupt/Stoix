#!/bin/bash
# stop CW-Duel stage A (run as a script file; do not inline pkill -f in an ssh command line)
OUT=/home/junweiluo/stoix_runs/cwduel_A2
date > $OUT/STOPPED
pkill -f "launch_cwduel_A2.sh"
pkill -f "system.algo_name=cwduel_r"
echo stopped
