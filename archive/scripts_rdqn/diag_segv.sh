#!/bin/bash
cd /home/junweiluo/Stoix; O=/home/junweiluo/stoix_runs/diag_segv; rm -rf $O; mkdir -p $O
for flags in "--xla_gpu_autotune_level=0" ""; do for env in HopperHop CheetahRun WalkerWalk; do
  tag=${env}_$( [ -n "$flags" ] && echo at0 || echo default )
  XLA_FLAGS="$flags" XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=1 timeout 900 .venv/bin/python scripts_rdqn/diag_segv.py $env > $O/$tag.log 2>&1; echo "$tag rc=$?" >> $O/progress.txt
done; done
JAX_PLATFORMS=cpu timeout 900 .venv/bin/python scripts_rdqn/diag_segv.py HopperHop > $O/HopperHop_cpu.log 2>&1; echo "HopperHop_cpu rc=$?" >> $O/progress.txt
date > $O/ALLDONE
