#!/bin/bash
# Stage 1 (2026-10-30): {DQN, RDQN} x {plain, CRN} x {asterix, freeway} x seeds {0,1,2}, 5M real env steps each.
# Freeway uses total_buffer_size=500000 (its 10x10x7 obs do not fit the default 1M buffer on a 24 GB GPU).
# Jobs whose jsonl already has {"done": true} are skipped.
# One job at a time per GPU (a job takes ~17 GB), 8 GPU workers, 3 jobs each. Each job retries up to 3x if it ends
# without the {"done": true} record (XLA GPU compile segfaults intermittently; autotune off + unlimited stack).
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/phase1; mkdir -p $OUT/_logs
git rev-parse HEAD > $OUT/COMMIT; git status --short > $OUT/GIT_STATUS; date >> $OUT/STARTED
JOBS=()
for g in asterix freeway; do for arm in dqn dqn_crn rdqn rdqn_crn; do for s in 0 1 2; do JOBS+=("$g $arm $s"); done; done; done
run_job() {
  local g=$1 arm=$2 s=$3 gpu=$4
  case $arm in
    dqn)      F="system.rdqn=False system.crn=False system.c_lr=0.0" ;;
    dqn_crn)  F="system.rdqn=False system.crn=True  system.c_lr=0.0" ;;
    rdqn)     F="system.rdqn=True  system.crn=False system.c_lr=1.0" ;;
    rdqn_crn) F="system.rdqn=True  system.crn=True  system.c_lr=1.0" ;;
  esac
  grep -q '"done": true' $OUT/${g}_${arm}_s${s}.jsonl 2>/dev/null && return
  local B=""; [ "$g" = freeway ] && B="system.total_buffer_size=500000"
  for attempt in 1 2 3; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 \
      XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/q_learning/ff_rdqn.py env=gymnax/$g arch.seed=$s \
        arch.total_timesteps=5e6 arch.num_evaluation=50 arch.absolute_metric=False \
        system.run_log_dir=$OUT $F $B > $OUT/_logs/${g}_${arm}_s${s}.a$attempt.log 2>&1 )
    if grep -q '"done": true' $OUT/${g}_${arm}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $g $arm s$s gpu$gpu attempt$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%T) FAIL $g $arm s$s gpu$gpu attempt$attempt" >> $OUT/_logs/progress.txt
  done
}
worker() {   # GPU $1 runs jobs $1, $1+8, $1+16 sequentially
  local gpu=$1
  for ((k=gpu; k<${#JOBS[@]}; k+=8)); do run_job ${JOBS[$k]} $gpu; done
}
for gpu in 0 1 2 3 4 5 6 7; do worker $gpu & sleep 2; done
wait
date > $OUT/ALLDONE
