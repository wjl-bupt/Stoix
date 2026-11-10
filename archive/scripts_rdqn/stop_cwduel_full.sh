#!/bin/bash
# run as a script file; never inline its pkill patterns in an ssh command line
OUT=/home/junweiluo/stoix_runs/cwduel_full
date > $OUT/STOPPED
pkill -f "launch_cwduel_full.sh"
pkill -f "system.algo_name=cwduel_mlp_k64"
echo stopped
