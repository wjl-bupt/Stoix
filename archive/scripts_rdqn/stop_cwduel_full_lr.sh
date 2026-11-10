#!/bin/bash
# run as a script file; never inline its pkill patterns in an ssh command line
OUT=/home/junweiluo/stoix_runs/cwduel_full_lr
date > $OUT/STOPPED
pkill -f "launch_cwduel_full_lr.sh"
pkill -f "system.algo_name=cwduel_lr32"
echo stopped
