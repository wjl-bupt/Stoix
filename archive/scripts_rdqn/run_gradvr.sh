#!/bin/bash
# usage: bash run_gradvr.sh <gpu> <worker> <n_workers>
cd /home/junweiluo/Stoix; RUN=/home/junweiluo/stoix_runs/diag_gap; mkdir -p $RUN/gradvr/_logs
J=(); for g in asterix seaquest; do for s in 0 1 2; do for t in 1499520 4998400; do J+=("$g $s $t"); done; done; done
GPU=$1
for ((k=$2; k<${#J[@]}; k+=$3)); do
  read g s t <<< "${J[$k]}"
  [ -f $RUN/gradvr/${g}_s${s}_t${t}.json ] && continue
  for att in 1 2 3; do
    ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python scripts_rdqn/grad_vr.py $RUN $g $s $t >> $RUN/gradvr/_logs/${g}_s${s}_t${t}.log 2>&1 )
    [ -f $RUN/gradvr/${g}_s${s}_t${t}.json ] && break
  done
done
echo "$(date) w$2 done" >> $RUN/gradvr/_logs/done.txt
