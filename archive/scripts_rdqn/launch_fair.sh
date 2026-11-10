#!/bin/bash
# Supplement to stage 1: budget-matched CRN.  CRN at 2.5M real steps = 5M env calls, epochs 32 (2x), batch 256
# pairs (= 512 transitions)  ->  same env calls, same number of gradient updates and same transitions per update
# as plain DQN/RDQN @5M (epochs 16, batch 512).  Starts after stage 1 (phase1/ALLDONE).
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/phase1_fair; mkdir -p $OUT/_logs
while [ ! -f /home/junweiluo/stoix_runs/phase1/ALLDONE ]; do sleep 30; done
git rev-parse HEAD > $OUT/COMMIT; date > $OUT/STARTED
JOBS=()
for g in asterix freeway; do for arm in dqn_crn rdqn_crn; do for s in 0 1 2; do JOBS+=("$g $arm $s"); done; done; done
run_job() {
  local g=$1 arm=$2 s=$3 gpu=$4
  grep -q '"done": true' $OUT/${g}_${arm}_s${s}.jsonl 2>/dev/null && return
  case $arm in
    dqn_crn)  F="system.rdqn=False system.crn=True system.c_lr=0.0" ;;
    rdqn_crn) F="system.rdqn=True  system.crn=True system.c_lr=1.0" ;;
  esac
  local B=""; [ "$g" = freeway ] && B="system.total_buffer_size=500000"
  for attempt in 1 2 3; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 \
      XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/q_learning/ff_rdqn.py env=gymnax/$g arch.seed=$s \
        arch.total_timesteps=2.5e6 arch.num_evaluation=25 arch.absolute_metric=False \
        system.epochs=32 system.total_batch_size=256 \
        system.run_log_dir=$OUT $F $B > $OUT/_logs/${g}_${arm}_s${s}.a$attempt.log 2>&1 )
    if grep -q '"done": true' $OUT/${g}_${arm}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $g $arm s$s gpu$gpu attempt$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%T) FAIL $g $arm s$s gpu$gpu attempt$attempt" >> $OUT/_logs/progress.txt
  done
}
worker() { local gpu=$1; for ((k=gpu; k<${#JOBS[@]}; k+=8)); do run_job ${JOBS[$k]} $gpu; done; }
for gpu in 0 1 2 3 4 5 6 7; do worker $gpu & sleep 2; done
wait
date > $OUT/ALLDONE
