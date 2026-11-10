#!/bin/bash
O=${OUT:-/home/junweiluo/stoix_runs/dmc_cwq}
echo "done: $(grep -l '"done"' $O/*.jsonl 2>/dev/null | wc -l)  running py: $(pgrep -f ff_cw_td3.py | wc -l)  fails: $(grep -c 'rc=[^0]' $O/progress.txt 2>/dev/null)"
for f in $O/*.jsonl; do [ -e "$f" ] || continue; n=$(($(wc -l < $f)-1)); l=$(tail -1 $f | cut -c1-90); echo "$(basename $f) n=$n $l"; done
