#!/bin/bash
# usage: bash run_gradattr.sh <gpu> <worker_idx> <n_workers>
cd /home/junweiluo/Stoix
RUN=/home/junweiluo/stoix_runs/diag_gap; mkdir -p $RUN/gradattr/_logs
J=()
for g in asterix seaquest; do for a in cwpair_mlp dqn; do for s in 0 1 2; do for t in 499840 1499520 2999040 4998400; do J+=("$g $a $s $t"); done; done; done; done
for ((k=$2; k<${#J[@]}; k+=$3)); do
  set -- ${J[$k]} $1 $2 $3; g=$1; a=$2; s=$3; t=$4
  [ -f $RUN/gradattr/${g}_${a}_s${s}_t${t}.json ] && { set -- $5 $6 $7; continue; }
  for att in 1 2 3 4; do
    ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=$5 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python scripts_rdqn/grad_attr.py $RUN $g $a $s $t >> $RUN/gradattr/_logs/${g}_${a}_s${s}_t${t}.log 2>&1 )
    [ -f $RUN/gradattr/${g}_${a}_s${s}_t${t}.json ] && break
  done
  set -- $5 $6 $7
done
echo "$(date) worker $2 done" >> $RUN/gradattr/_logs/workers_done.txt
