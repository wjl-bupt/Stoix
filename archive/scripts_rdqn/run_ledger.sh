#!/bin/bash
# usage: bash run_ledger.sh <gpu> <unit...>   unit = cwpair_mlp_s0 | dqn_s1 ...
cd /home/junweiluo/Stoix; R=/home/junweiluo/stoix_runs/ledger_20261103; GPU=$1; shift
for u in "$@"; do
  a=cw; [[ $u == dqn* ]] && a=dqn
  for w in $(seq 1 120); do grep -q '"done": true' $R/$u/*.jsonl 2>/dev/null && [ -f $R/$u/full_state.pkl ] && break; sleep 30; done
  for att in 1 2 3 4 5; do
    [ -f $R/ledger_$u/ledger.json ] && break
    ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python scripts_rdqn/decision_ledger.py $R/$u $a $R/ledger_$u > $R/ledger_$u.a$att.log 2>&1 )
  done
  echo "$(date +%F_%T) $u $( [ -f $R/ledger_$u/ledger.json ] && echo DONE || echo FAILED )" >> $R/ledger_progress.txt
done
