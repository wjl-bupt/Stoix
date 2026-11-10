#!/bin/bash
# 2026-11-03: same-trajectory pairing.  pgx MinAtar asterix + seaquest x seeds 0-4 x 5M, protocol = minatar_pgx.
# arms: cwpair_mlp_st (CW-Pair MLP, pairs only inside a trajectory, group 8 = 7 partners)
#       rdqn_st       (RDQN, centred loss restricted to same-trajectory pairs, group 8)
#       rdqn          (RDQN with the usual batch centring; pgx has no RDQN run yet -> needed as control)
# controls reused: minatar_pgx/{dqn,cwpair_mlp} seeds 0-4.
# Usage: GPU_LIST="1 3 0" NW=6 bash scripts_rdqn/launch_sametraj.sh   (skips DONE and running jobs; safe to restart)
cd /home/junweiluo/Stoix
OUT=${OUT:-/home/junweiluo/stoix_runs/same_traj}; mkdir -p $OUT/_logs $OUT/_code
cp stoix/systems/q_learning/ff_rdqn.py stoix/systems/q_learning/ff_rpair.py stoix/systems/q_learning/same_traj.py \
   stoix/systems/q_learning/alpal.py stoix/configs/system/q_learning/ff_rdqn.yaml scripts_rdqn/launch_sametraj.sh scripts_rdqn/check_same_traj.py $OUT/_code/
date >> $OUT/STARTED
GPUS=(${GPU_LIST:-1 3 0}); NW=${NW:-6}; TT=${TT:-5e6}; NE=${NE:-50}; SEEDS=${SEEDS:-0 1 2 3 4}; GAMES=${GAMES:-asterix seaquest}
ARMS=${ARMS:-cwpair_mlp_st rdqn_st rdqn}; AL=${AL:-0.9}
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=$TT arch.num_evaluation=$NE system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
cmd() {
  case $1 in
    rdqn)          echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0" ;;
    rdqn_st)       echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8" ;;
    rdqn_rg)       echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8 system.st_random=True" ;;
    rdqn_mix)      echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8 system.st_cross=7" ;;
    cwpair_mlp_mix) echo "stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.same_traj=True system.st_group=8 system.st_cross=7" ;;
    dqn_al)   echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=False system.c_lr=0.0 system.al_mode=al system.al_alpha=$AL" ;;
    dqn_pal)  echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=False system.c_lr=0.0 system.al_mode=pal system.al_alpha=$AL" ;;
    rdqn_st_al)  echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8 system.al_mode=al system.al_alpha=$AL" ;;
    rdqn_st_pal) echo "stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8 system.al_mode=pal system.al_alpha=$AL" ;;
    cwpair_mlp_mix_al)  echo "stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.same_traj=True system.st_group=8 system.st_cross=7 system.al_mode=al system.al_alpha=$AL" ;;
    cwpair_mlp_mix_pal) echo "stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.same_traj=True system.st_group=8 system.st_cross=7 system.al_mode=pal system.al_alpha=$AL" ;;
    cwpair_mlp_st) echo "stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.same_traj=True system.st_group=8" ;;
  esac
}
JOBS=()
for s in $SEEDS; do for g in $GAMES; do for a in $ARMS; do JOBS+=("$g $a $s"); done; done; done
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  pgrep -f "env=pgx/$g arch.seed=$s system.algo_name=$a$" > /dev/null && return
  for attempt in $(seq 1 30); do
    local t0=$(date +%s)
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python $(cmd $a) env=pgx/$g arch.seed=$s system.algo_name=$a > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    local t1=$(date +%s)
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%F_%T) DONE $g $a s$s gpu$gpu a$attempt wall_sec=$((t1-t0))" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%F_%T) FAIL $g $a s$s gpu$gpu a$attempt wall_sec=$((t1-t0))" >> $OUT/_logs/progress.txt
    rm -f $OUT/${g}_${a}_s${s}.jsonl
    sleep 60
  done
}
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 3; done
wait
date > $OUT/ALLDONE
