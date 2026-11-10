#!/bin/bash
# Status of a launcher run directory:  bash cwq/status.sh <OUT>
OUT=${1:?usage: status.sh <OUT>}
if [ -f $OUT/_launcher.pid ] && kill -0 $(cat $OUT/_launcher.pid) 2>/dev/null; then echo "launcher running (pid $(cat $OUT/_launcher.pid))"; else echo "launcher not running"; fi
ls $OUT/ALLDONE* 2>/dev/null | sed 's/^/finished: /'
n_done=$(for f in $OUT/*.jsonl; do [ -f "$f" ] && tail -n 1 "$f" | grep -q '"done"' && echo x; done | wc -l)
n_all=$(ls $OUT/*.jsonl 2>/dev/null | wc -l)
echo "jsonl files: $n_all, finished: $n_done"
if [ -f $OUT/_logs/progress.txt ]; then
  echo "progress.txt: $(grep -c ' DONE ' $OUT/_logs/progress.txt) DONE, $(grep -c ' FAIL ' $OUT/_logs/progress.txt) FAIL, $(grep -c GIVEUP $OUT/_logs/progress.txt) GIVEUP"
  echo "last lines:"; tail -n 5 $OUT/_logs/progress.txt; fi
echo "in progress (jsonl without done; last timestep):"
for f in $OUT/*.jsonl; do [ -f "$f" ] || continue; tail -n 1 "$f" | grep -q '"done"' && continue
  echo "  $(basename $f) $(tail -n 1 $f | grep -o '"timestep": [0-9]*\|"step": [0-9]*' | head -1)"; done
