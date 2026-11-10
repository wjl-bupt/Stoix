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
