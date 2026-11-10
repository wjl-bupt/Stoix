#!/bin/bash
OUT=/home/junweiluo/stoix_runs/nonterm
echo "done: $(grep -c DONE $OUT/_logs/progress.txt 2>/dev/null)/25  fail-attempts: $(grep -c FAIL $OUT/_logs/progress.txt 2>/dev/null)"
for f in $OUT/*_s*.jsonl; do [ -f "$f" ] || continue; n=$(grep -c '"timestep"' $f); last=$(grep '"timestep"' $f | tail -1 | grep -o '"eval_return": [0-9.e+-]*' | cut -d' ' -f2); sps=$(grep '"timestep"' $f | tail -1 | grep -o '"sps": [0-9.e+-]*' | cut -d' ' -f2); echo "$(basename $f) evals=$n/50 last_eval=$last sps=$sps $(grep -q '"done": true' $f && echo DONE)"; done
ls $OUT | grep -E "ALLDONE|STOPPED"
