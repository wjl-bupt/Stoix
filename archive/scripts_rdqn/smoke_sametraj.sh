#!/bin/bash
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/smoke_sametraj TT=3e5 NE=3 SEEDS=0 GAMES="asterix seaquest" GPU_LIST="1 3 1 3 0 0" NW=6 bash scripts_rdqn/launch_sametraj.sh
