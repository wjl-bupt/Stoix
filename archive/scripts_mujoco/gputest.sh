#!/bin/bash
cd /home/junweiluo/Stoix/scripts_mujoco; mkdir -p gputest; rm -f gputest/*
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0" KEEPALIVE=1 OMP_NUM_THREADS=1
PY=/home/junweiluo/Stoix/.venv/bin/python
go() { CUDA_VISIBLE_DEVICES=$1 $PY mujoco_offpolicy.py --env $2 --base $3 --head $4 --sampler $5 --seed $6 --total 40000 --eval_eps 2 --out gputest/g$1_$2_$3_$4_$6.jsonl > gputest/g$1_$2_$3_$4_$6.log 2>&1; echo "g$1 $2 $3 $4 $6 rc=$?" >> gputest/rc.txt; }
go 0 Hopper-v5 td3 q uniform 0 &
for i in $(seq 1 8); do go 1 Hopper-v5 td3 q uniform $i & done
for i in $(seq 1 8); do go 2 Hopper-v5 td3 q uniform $i & go 2 Humanoid-v5 sac q uniform $i & done
for i in $(seq 1 4); do go 3 Hopper-v5 td3 qcw rho $i & done
wait; touch gputest/ALLDONE
