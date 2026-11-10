#!/bin/bash
cd /home/junweiluo/Stoix; RUN=/home/junweiluo/stoix_runs/diag_gap
( ulimit -s unlimited; JAX_DEFAULT_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
  .venv/bin/python scripts_rdqn/grad_attr.py /tmp/ga_chk asterix cwpair_mlp 0 1499520 --check 2>&1 | grep CHECK )
