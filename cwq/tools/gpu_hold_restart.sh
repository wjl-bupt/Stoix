#!/bin/bash
# Restart only the holder MANAGER (holders keep their memory, the new manager adopts them).  Holders on GPUs that are no
# longer in GPUS are released.   GPUS="0 1 2" TARGET_MB=23552 bash cwq/tools/gpu_hold_restart.sh
B=/home/junweiluo/gpu_busy; D=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:-"0 1 2"}; TARGET_MB=${TARGET_MB:-23552}
[ -f $B/mgr.pid ] && kill $(cat $B/mgr.pid) 2>/dev/null; sleep 1
for f in $B/holder_*_*.pid; do [ -e "$f" ] || continue; g=$(basename $f | cut -d_ -f2)
  case " $GPUS " in *" $g "*) ;; *) kill -9 $(cat $f) 2>/dev/null; rm -f $f ${f%.pid}.mb; echo "$(date +%F_%T) release gpu $g (not in GPUS)" >> $B/log ;; esac; done
GPUS="$GPUS" TARGET_MB=$TARGET_MB setsid nohup bash $D/gpu_hold_mgr.sh > /dev/null 2>&1 < /dev/null &
sleep 2; echo "manager pid $(cat $B/mgr.pid), GPUS=\"$GPUS\" TARGET_MB=$TARGET_MB"
