#!/bin/bash
# Round 2 (2026-10-31): same-capacity DQN (sadqn) x 4 games; diagnostics breakout K=8 partners, freeway reset/K=8.
# Protocol identical to minatar_base (controls DQN/RDQN reused).  8 GPUs x 4 workers.
cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/minatar_pair2; mkdir -p $OUT/_logs
cp stoix/systems/q_learning/ff_rpair.py $OUT/ff_rpair.py.snapshot; cp scripts_rdqn/check_rpair.py $OUT/; cp scripts_rdqn/launch_rpair2.sh $OUT/; date >> $OUT/STARTED
P="env.wrapper=null arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=5e6 arch.num_evaluation=50"
GPUS=(0 1 2 3 4 5 6 7); NW=32
JOBS=()
for s in 0 1 2 3 4; do
  for g in asterix breakout freeway space_invaders; do JOBS+=("$g sadqn $s sadqn cross"); done
  JOBS+=("breakout pair_free_cross_K8 $s free cross"); JOBS+=("breakout pair_anti_cross_K8 $s anti cross")
  JOBS+=("freeway pair_free_cross_K8 $s free cross"); JOBS+=("freeway pair_free_cross_reset $s free cross")
done
extra() { case $1 in *_K8) echo "+system.n_partners=8" ;; *_reset) echo "system.reset_continuing=True" ;; *) echo "" ;; esac; }
run_job() {
  local g=$1 a=$2 s=$3 h=$4 t=$5 gpu=$6
  grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null && return
  for attempt in 1 2 3; do
    ( ulimit -s unlimited
      CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python stoix/systems/q_learning/ff_rpair.py $P env=gymnax/$g arch.seed=$s system.algo_name=$a \
        system.head=$h system.term=$t $(extra $a) system.run_log_dir=$OUT > $OUT/_logs/${g}_${a}_s${s}.a$attempt.log 2>&1 )
    if grep -q "\"done\": true" $OUT/${g}_${a}_s${s}.jsonl 2>/dev/null; then
      echo "$(date +%T) DONE $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt; return; fi
    echo "$(date +%T) FAIL $g $a s$s gpu$gpu a$attempt" >> $OUT/_logs/progress.txt
  done
}
worker() { local w=$1 gpu=${GPUS[$(( $1 % 8 ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run_job ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 1; done
wait
date > $OUT/ALLDONE
