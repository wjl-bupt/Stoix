#!/bin/bash
# run as a script file; never inline its pkill patterns in an ssh command line
OUT=/home/junweiluo/stoix_runs/cfpair
date > $OUT/STOPPED
pkill -f "launch_cfpair.sh"
pkill -f "system.algo_name=cwpair_mlp_(crn|ind)"
echo stopped
