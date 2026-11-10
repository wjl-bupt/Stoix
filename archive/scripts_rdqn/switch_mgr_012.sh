#!/bin/bash
# 2026-11-10 (user): hold GPU 0-2 only (release GPU3), target ~23.5 GB/card (reserve for DMControl later)
B=/home/junweiluo/gpu_busy; cd /home/junweiluo/Stoix/scripts_rdqn
kill $(cat $B/mgr.pid) 2>/dev/null; sleep 1
for f in $B/holder_3_*.pid $B/holder_3.pid; do [ -e "$f" ] || continue; kill -9 $(cat $f) 2>/dev/null; rm -f $f ${f%.pid}.mb; done
echo "$(date +%F_%T) release gpu 3 (user: hold 0-2 only)" >> $B/log
GPUS="0 1 2" TARGET_MB=23552 setsid nohup bash gpu_hold_mgr2.sh > /dev/null 2>&1 < /dev/null &
sleep 3; echo mgr $(cat $B/mgr.pid)
