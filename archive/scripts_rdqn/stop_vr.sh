#!/bin/bash
# run as a script file; never inline its pkill patterns in an ssh command line
OUT=/home/junweiluo/stoix_runs/vr
date > $OUT/STOPPED
pkill -f "launch_vr.sh"
pkill -f "system.algo_name=cwpair_mlp_(strat|multi|sm)"
echo stopped
