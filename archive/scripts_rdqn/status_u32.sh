#!/bin/bash
OUT=${OUT:-/home/junweiluo/stoix_runs/minatar_u32}
echo "done: $(grep -l '"done": true' $OUT/*.jsonl 2>/dev/null | wc -l)   running: $(pgrep -fc 'system.algo_name=(dqn|ddqn|dueling|c51|rdqn_rho|cwpair_rho|cwduel_rho)$')"
tail -4 $OUT/_logs/progress.txt 2>/dev/null
