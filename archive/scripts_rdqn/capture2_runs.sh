#!/bin/bash
# 2026-11-03 g-score checkpoints: {dqn, rdqn, rdqn_st, cwpair_mlp, cwpair_mlp_st} x {asterix, seaquest} x seeds 0-1, 2M real steps,
# protocol = minatar_pgx, full learner state saved at the end.  usage: GPU_LIST="0 1 2" NW=6 bash scripts_rdqn/capture2_runs.sh
cd /home/junweiluo/Stoix; R=/home/junweiluo/stoix_runs/gscore_20261103; mkdir -p $R
P0="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=2e6 arch.num_evaluation=20"
P="$P0 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
cmd() { case $1 in
  dqn)  echo "capture2_ff_rdqn.py $P network=minatar_dqn system.rdqn=False system.c_lr=0.0" ;;
  rdqn) echo "capture2_ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0" ;;
  rdqn_st) echo "capture2_ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8" ;;
  cwpair_mlp) echo "capture2_ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8" ;;
  cwpair_mlp_st) echo "capture2_ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.same_traj=True system.st_group=8" ;;
esac; }
GPUS=(${GPU_LIST:-0 1 2}); NW=${NW:-6}
JOBS=(); for s in 0 1; do for g in asterix seaquest; do for a in dqn rdqn rdqn_st cwpair_mlp cwpair_mlp_st; do JOBS+=("$g $a $s"); done; done; done
run() { local g=$1 a=$2 s=$3 gpu=$4; local D=$R/${g}_${a}_s$s; mkdir -p $D
  for att in $(seq 1 20); do
    [ -f $D/full_state.pkl ] && grep -q '"done": true' $D/*.jsonl 2>/dev/null && return
    ( ulimit -s unlimited; CUDA_VISIBLE_DEVICES=$gpu XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0 \
      .venv/bin/python scripts_rdqn/$(cmd $a) env=pgx/$g arch.seed=$s system.algo_name=$a system.run_log_dir=$D > $D/capture.a$att.log 2>&1 )
    echo "$(date +%F_%T) $g $a s$s gpu$gpu attempt $att exit $?" >> $R/capture_progress.txt; sleep 20
  done; }
worker() { local w=$1 gpu=${GPUS[$(( $1 % ${#GPUS[@]} ))]}; for ((k=w; k<${#JOBS[@]}; k+=NW)); do run ${JOBS[$k]} $gpu; done; }
for w in $(seq 0 $((NW-1))); do worker $w & sleep 3; done; wait; date > $R/CAPTURE_DONE
