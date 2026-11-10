#!/bin/bash
# 2026-11-05: which XLA flag removes the GPU compile segfault of scan(MJX step)?  HopperHop, 3 repeats per variant.
cd /home/junweiluo/Stoix; O=/home/junweiluo/stoix_runs/diag_flags; rm -rf $O; mkdir -p $O
V0="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1"
V1="--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer="
V2="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false"
V3="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer="
var() { local i=$1 fl=$2
  for r in 1 2 3; do XLA_FLAGS="$fl" XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$i timeout 900 .venv/bin/python scripts_rdqn/diag_segv.py HopperHop > $O/V${i}_$r.log 2>&1
    echo "V$i r$r rc=$?" >> $O/progress.txt; done; }
var 0 "$V0" & var 1 "$V1" & var 2 "$V2" & var 3 "$V3" &
wait; date > $O/ALLDONE
