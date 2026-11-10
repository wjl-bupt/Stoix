#!/bin/bash
# 2026-11-10 switch v1 (GPU0-3, frac) -> v2 (GPU0-1, 22 GB target); free GPU2/3 completely
cd /home/junweiluo/Stoix/scripts_rdqn; B=/home/junweiluo/gpu_busy
cp gpu_busy.sh gpu_busy.sh.bak_20261110_before_mgr2
cat > gpu_busy.sh <<'EOT'
# 2026-11-08 (v2 2026-11-10)  source in launchers.  gpu_busy_on <gpu>: mark busy (flag owned by this shell's $BASHPID) and
# kill the manager's holders on that GPU (holder_<gpu>.pid of v1 and holder_<gpu>_<n>.pid of v2).  gpu_busy_off <gpu>: drop
# the flag (manager re-holds when no flag is left).  Since 2026-11-10 we use GPU 0 and 1 only.
GB=/home/junweiluo/gpu_busy; mkdir -p $GB
gpu_busy_on() { local g=$1; touch $GB/busy_${g}_$BASHPID
  for i in $(seq 1 60); do local left=0
    for f in $GB/holder_${g}.pid $GB/holder_${g}_*.pid; do [ -e "$f" ] || continue; local p=$(cat $f)
      if [ -n "$p" ] && kill -0 $p 2>/dev/null; then kill -9 $p; left=1; fi; done
    if [ $left = 1 ]; then sleep 2; else [ $i -gt 3 ] && break; sleep 1; fi; done; }
gpu_busy_off() { rm -f $GB/busy_${1}_$BASHPID; }
EOT
kill $(cat $B/mgr.pid); sleep 1
for g in 2 3; do kill -9 $(cat $B/holder_$g.pid) 2>/dev/null; rm -f $B/holder_$g.pid $B/held_$g; echo "$(date +%F_%T) release gpu $g (v1 retired; GPU 2-7 left to others)" >> $B/log; done
for g in 0 1; do mv $B/holder_$g.pid $B/holder_${g}_0.pid; echo 2990 > $B/holder_${g}_0.mb; done   # adopt v1 holders (2990 MB each)
setsid nohup bash gpu_hold_mgr2.sh > /dev/null 2>&1 < /dev/null &
sleep 3; echo mgr2 $(cat $B/mgr.pid)
