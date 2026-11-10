#!/bin/bash
# PPO vs R-PPO (2026-11-01): {asterix, breakout} x {ppo, rppo} x seeds 0-4 x 10M env steps (gymnax MinAtar).
# Protocol = Stoix ff_ppo defaults (lr 3e-4 linear decay, rollout 128, 4 epochs x 16 minibatches, gamma .99,
# lambda .95, clip .2 incl. value clipping, ent .01, vf .5, grad-norm .5, adv standardisation) except:
# 128 envs (1024 gives only 76 updates in 10M), MinAtar CNN torso for actor & critic (separate), no flatten wrapper.
# R-PPO: V = v_theta + c, centred network loss, c <- c + 1.0 * mean residual per minibatch (SGD).
# A job is done iff its jsonl ends with {"done": true}; up to 3 attempts (XLA segfaults).
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/minatar_rppo_10m; mkdir -p $OUT/_logs
git rev-parse HEAD > $OUT/COMMIT; cp stoix/systems/ppo/anakin/ff_rppo.py $OUT/ff_rppo.py.snapshot; date >> $OUT/STARTED
P="env.wrapper=null network=minatar_ppo arch.total_num_envs=128 arch.total_timesteps=1e7 arch.num_evaluation=50 arch.num_eval_episodes=128 arch.absolute_metric=False"
GPUS=(${GPUS:-0 2 3 4 5})
JOBS=()
for s in 0 1 2 3 4; do for g in asterix breakout; do for a in ppo rppo; do JOBS+=("$g $a $s"); done; done; done
run_job() {
  local g=$1 a=$2 s=$3 gpu=$4
  grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  case $a in
    ppo)  F="system.rppo=False system.c_lr=0.0" ;;
    rppo) F="system.rppo=True system.c_lr=1.0" ;;
  esac
  for attempt in 1 2 3; do
    [ -f $OUT/STOPPED ] && return
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/ppo/anakin/ff_rppo.py $P env=gymnax/$g arch.seed=$s system.algo_name=$a \
        system.run_log_dir=$OUT $F > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q '"done": true' $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
  done
}
NW=${NW:-20}
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 2; done
wait
date > $OUT/ALLDONE
