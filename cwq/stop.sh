#!/bin/bash
# Stop a launcher and all runs it started:  bash cwq/stop.sh <OUT>
# Kills the launcher's process group (workers + python runs).  Unfinished jsonl files stay (they are re-run on restart).
OUT=${1:?usage: stop.sh <OUT>}
P=$OUT/_launcher.pid
[ -f $P ] || { echo "no $P"; exit 1; }
pid=$(cat $P); pg=$(ps -o pgid= -p $pid 2>/dev/null | tr -d ' ')
[ -n "$pg" ] || { echo "launcher $pid not running"; rm -f $P; exit 0; }
if [ "$pg" = "$pid" ]; then kill -TERM -- -$pg; sleep 3; kill -KILL -- -$pg 2>/dev/null      # detached launcher = own group
else  # launcher run with FG=1 inside someone's shell: kill only its process tree
  tree() { local p=$1; for c in $(pgrep -P $p); do tree $c; done; echo $p; }
  pids=$(tree $pid); kill -TERM $pids 2>/dev/null; sleep 3; kill -KILL $pids 2>/dev/null; fi
rm -f $P; date > $OUT/STOPPED_$(date +%Y%m%d_%H%M); echo "stopped launcher $pid (process group $pg)"
