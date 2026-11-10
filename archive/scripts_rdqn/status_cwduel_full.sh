#!/bin/bash
OUT=/home/junweiluo/stoix_runs/cwduel_full
echo "done: $(grep -c DONE $OUT/_logs/progress.txt 2>/dev/null)/50  fail-attempts: $(grep -c FAIL $OUT/_logs/progress.txt 2>/dev/null)"
for f in $OUT/*_s*.jsonl; do [ -f "$f" ] || continue; n=$(grep -c '"timestep"' $f); last=$(grep '"timestep"' $f | tail -1 | grep -o '"eval_return": [0-9.e+-]*' | cut -d' ' -f2); echo "$(basename $f) evals=$n/50 last_eval=$last $(grep -q '"done": true' $f && echo DONE)"; done
ls $OUT | grep -E "ALLDONE|STOPPED"
