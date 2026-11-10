#!/bin/bash
# run as a script file; never inline its pkill patterns in an ssh command line
OUT=/home/junweiluo/stoix_runs/nonterm
date > $OUT/STOPPED
pkill -f "launch_nonterm.sh"
pkill -f "system.algo_name=cwpair_mlp_nn"
echo stopped
