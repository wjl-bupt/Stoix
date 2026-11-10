#!/bin/bash
# usage: bash run_d3.sh <gpu> <worker> <n_workers>
cd /home/junweiluo/Stoix; RUN=/home/junweiluo/stoix_runs/diag_gap; mkdir -p $RUN/d3/_logs
J=(); for t in 4998400 2999040 1499520; do for g in asterix seaquest; do for s in 0 1 2; do J+=("$g $s $t"); done; done; done
for ((k=$2; k<${#J[@]}; k+=$3)); do
  read g s t <<< "${J[$k]}"
  [ -f $RUN/d3/${g}_s${s}_t${t}.json ] && continue
  for att in 1 2 3 4; do
    ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=$1 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python scripts_rdqn/d3_readout.py $RUN $g $s $t >> $RUN/d3/_logs/${g}_s${s}_t${t}.log 2>&1 )
    [ -f $RUN/d3/${g}_s${s}_t${t}.json ] && break
  done
done
echo "$(date) w$2 done" >> $RUN/d3/_logs/done.txt
