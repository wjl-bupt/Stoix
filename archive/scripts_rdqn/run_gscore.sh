#!/bin/bash
# run gscore.py on every captured state in gscore_20261103 (skips done); loops until CAPTURE_DONE and all processed
cd /home/junweiluo/Stoix; R=/home/junweiluo/stoix_runs/gscore_20261103; GPU=${GPU:-1}; mkdir -p $R/results
while true; do
  for D in $R/*_s[0-9]; do
    n=$(basename $D); a=$(echo $n | sed -E 's/^(asterix|seaquest)_//; s/_s[0-9]$//')
    case $a in cwpair*) k=cw ;; *) k=dqn ;; esac
    [ -f $R/results/$n.json ] && continue
    [ -f $D/full_state.pkl ] && grep -q '"done": true' $D/*.jsonl 2>/dev/null || continue
    [ -f $R/results/$n.lock ] && continue; touch $R/results/$n.lock
    CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python scripts_rdqn/gscore.py $D $k $R/results/$n.json > $R/results/$n.log 2>&1 || rm -f $R/results/$n.json
    rm -f $R/results/$n.lock
  done
  if [ -f $R/CAPTURE_DONE ] && [ $(ls $R/results/*.json 2>/dev/null | wc -l) -ge 20 ]; then date > $R/GSCORE_DONE; break; fi
  sleep 60
done
