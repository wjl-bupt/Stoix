#!/bin/bash
# DIAG 2026-11-01: asterix/seaquest x {dqn, cwpair_mlp, cwpair_lr} x seeds 0-2 x 5M, same protocol as the main pgx run
# (launch_minatar_pgx.sh) + pre-clip global grad norm logging (system.diag_grad) + parameter checkpoints every 0.5M
# (system.ckpt_every=5 evaluations).  Diag copies ff_rdqn_diag.py / ff_rpair_diag.py; the main-run files are untouched.
# usage: GPU_LIST="6 7" NW=8 bash launch_diag.sh
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/diag_gap; mkdir -p $OUT/_logs $OUT/_code
:
date >> $OUT/STARTED_extra
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50 system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 +system.diag_grad=True +system.ckpt_every=5"
GPUS=(${GPU_LIST:-6 7}); NW=${NW:-8}
ALGOS="dqn cwpair_mlp cwpair_lr"
JOBS=()
IFS=";" read -ra JOBS <<< "${JOBLIST:?}"
cmd() {
  local g=$1 a=$2 s=$3
  case $a in
    dqn)        echo "stoix/systems/q_learning/ff_rdqn_diag.py $P network=minatar_dqn system.rdqn=False system.c_lr=0.0" ;;
    cwpair_mlp) echo "stoix/systems/q_learning/ff_rpair_diag.py $P system.head=free system.term=cross +system.n_partners=8" ;;
    cwpair_lr)  echo "stoix/systems/q_learning/ff_rpair_diag.py $P system.head=lowrank system.term=cross +system.all_pairs=True +system.pair_rank=32" ;;
  esac
}
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  for attempt in 1 2 3 4 5 6; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python $(cmd $g $a $s) env=pgx/$g arch.seed=$s system.algo_name=$a > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%F_%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%F_%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
  done
}
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 2; done
wait
date > $OUT/EXTRADONE
