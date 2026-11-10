#!/bin/bash
OUT=/home/junweiluo/stoix_runs/minatar_rppo_10m; touch $OUT/STOPPED
pkill -f "launch_rppo.sh"; pkill -f "ff_rppo.py.*minatar_rppo_10m"; echo stopped
