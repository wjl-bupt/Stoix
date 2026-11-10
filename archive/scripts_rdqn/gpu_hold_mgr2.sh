#!/bin/bash
# 2026-11-10 GPU holder manager v2 (user): we only use GPU 0 and 1 (2-7 left to others).  When none of our experiments
# uses a GPU, hold TARGET_MB (default 22 GB) of it.  Other users' memory is never touched: if the GPU is not free enough,
# hold what is free (minus a 512 MB margin) and add further holders whenever more memory becomes free, until TARGET_MB.
# Holders: $B/holder_<gpu>_<n>.pid with size in $B/holder_<gpu>_<n>.mb.  Busy flags as before ($B/busy_<gpu>_<pid>).
# Start: setsid nohup bash gpu_hold_mgr2.sh > /dev/null 2>&1 < /dev/null &     Stop: bash gpu_hold_stop.sh
B=/home/junweiluo/gpu_busy; mkdir -p $B
GPUS=${GPUS:-"0 1"}; TARGET_MB=${TARGET_MB:-22528}; STEP_MIN=${STEP_MIN:-1024}
PY=/home/junweiluo/Stoix/.venv/bin/python; HOLD=/home/junweiluo/Stoix/scripts_rdqn/gpu_hold.py
echo $$ > $B/mgr.pid
alive() { [ -n "$1" ] && kill -0 $1 2>/dev/null; }
busy() { local g=$1 any=1
  for f in $B/busy_${g}_*; do [ -e "$f" ] || continue
    if alive ${f##*_}; then any=0; else rm -f $f; fi; done; return $any; }
held_mb() { local g=$1 s=0
  for f in $B/holder_${g}_*.pid; do [ -e "$f" ] || continue
    if alive $(cat $f); then s=$((s + $(cat ${f%.pid}.mb))); else rm -f $f ${f%.pid}.mb; fi; done; echo $s; }
release() { local g=$1 any=0
  for f in $B/holder_${g}_*.pid; do [ -e "$f" ] || continue; kill -9 $(cat $f) 2>/dev/null; rm -f $f ${f%.pid}.mb; any=1; done
  [ $any = 1 ] && echo "$(date +%F_%T) release gpu $g" >> $B/log; }
hold_more() { local g=$1 have=$(held_mb $1)
  local tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $g)
  local fr=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
  local want=$((TARGET_MB - have)); local can=$((fr - 512 - 300))            # 300 MB: CUDA context of the new holder
  [ $can -lt $want ] && want=$can
  [ $want -lt $STEP_MIN ] && return
  local f=$(awk -v w=$want -v t=$tot 'BEGIN{printf "%.4f", w/t}')
  local n=$(date +%s%N)
  CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=$f \
    setsid nohup $PY $HOLD $B/held_${g}_$n >> $B/holder_$g.log 2>&1 < /dev/null &
  echo $! > $B/holder_${g}_$n.pid; echo $want > $B/holder_${g}_$n.mb
  echo "$(date +%F_%T) hold gpu $g +${want}MB (had ${have}MB, free ${fr}MB)" >> $B/log; sleep 20; }
while true; do
  for g in $GPUS; do
    if busy $g; then release $g; else hold_more $g; fi
  done
  sleep 10
done
