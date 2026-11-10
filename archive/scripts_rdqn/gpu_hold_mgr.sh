#!/bin/bash
# 2026-11-08 GPU holder manager (user rule): while OUR experiments run on a GPU, no holder there; when none of our
# experiments uses the GPU, hold ~80% of its memory (less if other users already take memory: frac = min(0.80, free-0.5GB)).
# Launchers mark a GPU busy with files $B/busy_<gpu>_<pid> (pid = the worker shell); stale flags (dead pid) are removed.
# Start once:  setsid nohup bash gpu_hold_mgr.sh > /dev/null 2>&1 < /dev/null &      Stop: bash gpu_hold_stop.sh
B=/home/junweiluo/gpu_busy; mkdir -p $B
GPUS=${GPUS:-"0 1 2 3"}; FRAC=${FRAC:-0.80}
PY=/home/junweiluo/Stoix/.venv/bin/python; HOLD=/home/junweiluo/Stoix/scripts_rdqn/gpu_hold.py
echo $$ > $B/mgr.pid
alive() { [ -n "$1" ] && kill -0 $1 2>/dev/null; }
busy() { local g=$1 any=1
  for f in $B/busy_${g}_*; do [ -e "$f" ] || continue
    if alive ${f##*_}; then any=0; else rm -f $f; fi; done; return $any; }
hpid() { cat $B/holder_$1.pid 2>/dev/null; }
release() { local p=$(hpid $1); alive $p && { kill -9 $p; echo "$(date +%F_%T) release gpu $1" >> $B/log; }; rm -f $B/holder_$1.pid $B/held_$1; }
hold() { local g=$1
  local tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $g)
  local fr=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
  local f=$(awk -v t=$tot -v fr=$fr -v F=$FRAC 'BEGIN{x=(fr-512)/t; if(x>F)x=F; if(x<0.05)x=0; printf "%.3f", x}')
  [ "$f" = "0.000" ] && return
  rm -f $B/held_$g
  CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=$f \
    setsid nohup $PY $HOLD $B/held_$g >> $B/holder_$g.log 2>&1 < /dev/null &
  echo $! > $B/holder_$g.pid; echo "$(date +%F_%T) hold gpu $g frac $f" >> $B/log; sleep 20; }
while true; do
  for g in $GPUS; do
    if busy $g; then release $g
    else alive $(hpid $g) || hold $g; fi
  done
  sleep 10
done
