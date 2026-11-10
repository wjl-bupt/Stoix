#!/bin/bash
# =====================================================================================================================
# Gym MuJoCo (Gymnasium v5) launcher: CW-Q-rho family + baselines.   bash cwq/run_mujoco.sh
# Implementation: cwq/mujoco_gym/mujoco_offpolicy.py (single-file JAX, not a Stoix system).
#
# Protocol (= CleanRL defaults, Fujimoto et al. 2018 / Haarnoja et al. 2018; fixed 11-07):
#   1 env, 1M env steps, UTD 1, batch 256, replay 1M, gamma .99, tau .005, MLP 256x256 ReLU, Adam (default eps 1e-8),
#   no obs/reward normalisation, no grad clip.  TD3/DDPG lr 3e-4, 25k random warm-up, expl noise .1;
#   TD3 target smoothing .2 clip .5, policy delay 2; DDPG = one critic, no smoothing; SAC q-lr 1e-3, pi-lr 3e-4, auto alpha.
#   Time-limit truncation is bootstrapped (only true failures terminate).  Eval every 5000 steps, 10 deterministic episodes.
#   5 envs x seeds 0-4.
#
# Algorithms (ALGOS):
#   ours       cwq_td3     CW-Q-rho on TD3 with the level-free (batch-centred) min      [old arm cwq_cmin]
#              cwq_td3raw  CW-Q-rho on TD3 with the raw min                            [old arm cwq_rho]
#              cwq_ddpg    CW-Q-rho on DDPG                                            [old arm cwq_ddpg]
#   baselines  td3 ddpg sac
#   default    ALGOS="cwq_td3 cwq_ddpg td3 ddpg sac"
# Knobs of OUR algorithms: RHO=0.5 G=8 NONORM=1 (0 = old xi/(1-gamma); suffix _norm)   EXTRA_OURS="--flag v" (needs TAG)
# Protocol knobs: TOTAL=1000000 (needs TAG if changed)   EXTRA_ALL (needs TAG)    CKPT=0 (1 = save nets at 20..100%)
# Selection / resources:
#   ENVS="Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5"  SEEDS="0 1 2 3 4"
#   GPU_LIST="0 1 2"  NW_PER_GPU=8 (each run is small; ~240 sps at 8 runs/GPU)  RETRIES=3  OUT=$RUNS_ROOT/mujoco_main
# Run file: $OUT/<Env>_<algo><suffix><_TAG>_s<seed>.jsonl    (last line {"done": ...})
# =====================================================================================================================
RETRIES=${RETRIES:-3}
CWQ_LAUNCHER=$(readlink -f "$0"); source "$(dirname "$CWQ_LAUNCHER")/lib/common.sh"; cd "$CWQ_DIR/mujoco_gym"
BENCH=mujoco
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_autotune_level=0 KEEPALIVE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ALGOS=${ALGOS:-"cwq_td3 cwq_ddpg td3 ddpg sac"}
ENVS=${ENVS:-"Humanoid-v5 Ant-v5 Walker2d-v5 Hopper-v5 HalfCheetah-v5"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
NW_PER_GPU=${NW_PER_GPU:-8}
OUT=${OUT:-$RUNS_ROOT/mujoco_main}
RHO_DEFAULT=0.5; G_DEFAULT=8; RHO=${RHO:-$RHO_DEFAULT}; G=${G:-$G_DEFAULT}; NONORM=${NONORM:-1}
TOTAL=${TOTAL:-1000000}; CKPT=${CKPT:-0}; TAG=${TAG:-}
[ "$TOTAL" != 1000000 ] && cwq_require_tag_if "TOTAL"
[ -n "${EXTRA_ALL:-}${EXTRA_OURS:-}" ] && cwq_require_tag_if "EXTRA_ALL/EXTRA_OURS"
CODE_SNAPSHOT=(cwq/mujoco_gym/mujoco_offpolicy.py)
OURS="--head qcw --sampler rho --rho $RHO --G $G --no_norm $NONORM"
SUF=$(cwq_ours_suffix); TSUF=${TAG:+_$TAG}

is_ours() { case $1 in cwq_td3|cwq_td3raw|cwq_ddpg) return 0 ;; *) return 1 ;; esac; }
run_name() { if is_ours $1; then echo "$1$SUF$TSUF"; else echo "$1$TSUF"; fi; }
algo_args() {
  case $1 in
    cwq_td3)    echo "--base td3 $OURS --min_mode centered ${EXTRA_OURS:-}" ;;
    cwq_td3raw) echo "--base td3 $OURS ${EXTRA_OURS:-}" ;;
    cwq_ddpg)   echo "--base ddpg $OURS ${EXTRA_OURS:-}" ;;
    td3)        echo "--base td3 --head q --sampler uniform" ;;
    ddpg)       echo "--base ddpg --head q --sampler uniform" ;;
    sac)        echo "--base sac --head q --sampler uniform" ;;
    *) echo "unknown algorithm $1" >&2; exit 1 ;;
  esac
}
job_out() { echo "$OUT/${1}_$(run_name $2)_s$3.jsonl"; }
job_cmd() { local ck=""; [ "$CKPT" = 1 ] && ck="--ckpt_dir $OUT/ckpt"
  echo "$PY mujoco_offpolicy.py --env $1 --seed $3 $(algo_args $2) --total $TOTAL ${EXTRA_ALL:-} $ck --out $(job_out $1 $2 $3)"; }
job_env() { :; }
for a in $ALGOS; do algo_args $a > /dev/null || exit 1; done
JOBS=(); for e in $ENVS; do for s in $SEEDS; do for a in $ALGOS; do JOBS+=("$e $a $s"); done; done; done   # slow envs first
cwq_main "$@"
