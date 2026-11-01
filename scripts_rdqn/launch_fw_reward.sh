#!/bin/bash
# Freeway re-run with rollout_reward_per_step logged (training-reward curve); same configs as phase1 / phase1_fair.
cd /home/junweiluo/Stoix
OUT1=/home/junweiluo/stoix_runs/phase1_freeway_r; OUTF=/home/junweiluo/stoix_runs/phase1_fair_freeway_r
mkdir -p $OUT1/_logs $OUTF/_logs
JOBS=()
for arm in dqn dqn_crn rdqn rdqn_crn; do for s in 0 1 2; do JOBS+=("p1 $arm $s"); done; done
for arm in dqn_crn rdqn_crn; do for s in 0 1 2; do JOBS+=("fair $arm $s"); done; done
run_job() {
  local kind=$1 arm=$2 s=$3 gpu=$4 OUT EXTRA
  if [ $kind = p1 ]; then OUT=$OUT1; EXTRA="arch.total_timesteps=5e6 arch.num_evaluation=50"
  else OUT=$OUTF; EXTRA="arch.total_timesteps=2.5e6 arch.num_evaluation=25 system.epochs=32 system.total_batch_size=256"; fi
  grep -q '"done": true' $OUT/freeway_${arm}_s${s}.jsonl 2>/dev/null && return
  case $arm in
    dqn)      F="system.rdqn=False system.crn=False system.c_lr=0.0" ;;
    dqn_crn)  F="system.rdqn=False system.crn=True  system.c_lr=0.0" ;;
    rdqn)     F="system.rdqn=True  system.crn=False system.c_lr=1.0" ;;
    rdqn_crn) F="system.rdqn=True  system.crn=True  system.c_lr=1.0" ;;
  esac
  for attempt in 1 2 3; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/q_learning/ff_rdqn.py env=gymnax/freeway arch.seed=$s arch.absolute_metric=False \
        system.total_buffer_size=500000 $EXTRA system.run_log_dir=$OUT $F > $OUT/_logs/freeway_${arm}_s${s}.a$attempt.log 2>&1 )
    if grep -q '"done": true' $OUT/freeway_${arm}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $kind $arm s$s gpu$gpu" >> /home/junweiluo/stoix_runs/fw_progress.txt; return; fi
    echo "$(date +%T) FAIL $kind $arm s$s gpu$gpu a$attempt" >> /home/junweiluo/stoix_runs/fw_progress.txt
  done
}
worker() { local gpu=$1; for ((k=gpu; k<${#JOBS[@]}; k+=8)); do run_job ${JOBS[$k]} $gpu; done; }
for gpu in 0 1 2 3 4 5 6 7; do worker $gpu & sleep 2; done
wait
date > /home/junweiluo/stoix_runs/fw_ALLDONE
