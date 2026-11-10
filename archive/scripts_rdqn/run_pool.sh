#!/bin/bash
# 2026-11-04 job pool under ONE gpu_guard (GPU0-3): runs $INIT, then every 60 s starts any new $OUT/_queue/*.sh
# (renamed *.started); exits when nothing is running and nothing is queued (guard then restarts the placeholder).
cd /home/junweiluo/Stoix
OUT=${OUT:-/home/junweiluo/stoix_runs/minatar_ep8_more}; Q=$OUT/_queue; mkdir -p $Q
bash -c "$INIT" &
sleep 5
while true; do
  for f in $Q/*.sh; do [ -e "$f" ] || continue; mv "$f" "$f.started"; bash "$f.started" & done
  sleep 60
  [ -z "$(jobs -rp)" ] && [ -z "$(ls $Q/*.sh 2>/dev/null)" ] && break
done
wait
