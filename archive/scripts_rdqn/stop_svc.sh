#!/bin/bash
# run as a script file; never inline its pkill patterns in an ssh command line
OUT=/home/junweiluo/stoix_runs/strat_vs_cw
date > $OUT/STOPPED
pkill -f "launch_svc.sh"
pkill -f "run_log_dir=/home/junweiluo/stoix_runs/strat_vs_cw"
echo stopped
