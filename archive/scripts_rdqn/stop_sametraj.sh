#!/bin/bash
# run as a script file (pkill pattern must not appear in the ssh command line)
OUT=${OUT:-/home/junweiluo/stoix_runs/same_traj}
pkill -f "launch_sametraj.sh"; sleep 1
pkill -f 'system.algo_name=(rdqn|rdqn_st|rdqn_rg|rdqn_mix|cwpair_mlp_st|cwpair_mlp_mix|dqn_al|dqn_pal|rdqn_st_al|rdqn_st_pal|cwpair_mlp_mix_al|cwpair_mlp_mix_pal)$'
date > $OUT/STOPPED
