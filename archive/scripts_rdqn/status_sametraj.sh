#!/bin/bash
OUT=${OUT:-/home/junweiluo/stoix_runs/same_traj}
echo "done: $(grep -l '"done": true' $OUT/*.jsonl 2>/dev/null | wc -l)  running: $(pgrep -fc 'system.algo_name=(rdqn|rdqn_st|rdqn_rg|rdqn_mix|cwpair_mlp_st|cwpair_mlp_mix|dqn_al|dqn_pal|rdqn_st_al|rdqn_st_pal|cwpair_mlp_mix_al|cwpair_mlp_mix_pal)$')"
/home/junweiluo/Stoix/.venv/bin/python /home/junweiluo/Stoix/scripts_rdqn/status_sametraj.py $OUT
tail -3 $OUT/_logs/progress.txt 2>/dev/null
