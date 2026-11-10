#!/bin/bash
# =====================================================================================================================
# MinAtar launcher: CW-Q-rho family + baselines.   bash cwq/run_minatar.sh          (detaches; see cwq/README.md)
#
# Protocol (default = our current main protocol, 2026-11-10):
#   pgx MinAtar (sticky actions 0.1, episode cap 10000), 5 games, seeds 0-9, 5M env steps, 128 envs, rollout 1,
#   8 gradient updates per vector step (epochs 8), batch 512, replay 100k, Adam lr 5e-4 (eps 1e-5), grad clip 10,
#   eps-greedy 1 -> .1 over 100k steps, target net every 1000 updates, warmup 40 vector steps, gamma .99,
#   50 greedy evaluations x 128 episodes.  Checkpoints at 20/40/60/80/100% (CKPT=1).
#
# Algorithms (ALGOS, space separated):
#   ours       cwq        CW-Q-rho: plain Q head, all-pairs cross-weighted pair loss, rho pair kernel over
#                         same-trajectory groups of G items, NO 1/(1-gamma) normalisation (NONORM=1, default since 11-10)
#              cwq_duel   CW-Duel-rho (dueling head V+A, rank 0), same loss / kernel
#   baselines  dqn ddqn dueling c51 rc51 rc       (pqn available, not in the default list)
#   default    ALGOS="cwq dqn ddqn dueling c51 rc51 rc"
#
# Knobs of OUR algorithms (non-default values are appended to the run name, e.g. cwq_rho0.0_g2_norm):
#   RHO=0.5  cross-trajectory pair mass      G=8  same-trajectory group size      NONORM=1 (0 = old xi/(1-gamma))
#   NSTEP=1  n-step look-ahead               RHO_CROSS_K=0  sparse cross-trajectory partners (0 = whole batch)
#   EXTRA_OURS="hydra overrides"  added to our algorithms only        (requires TAG=...)
# Protocol knobs (all algorithms; changing any of them requires TAG=..., which is appended to every run name):
#   TT=5e6 NUM_EVAL=50 EPOCHS=8 LR=5e-4 BATCH=512      EXTRA_ALL="hydra overrides"
# Selection / resources:
#   GAMES="asterix breakout freeway seaquest space_invaders"  SEEDS="0 1 2 3 4 5 6 7 8 9"
#   GPU_LIST="0 1 2"  NW_PER_GPU=3  RETRIES=30  OUT=$RUNS_ROOT/minatar_main  CKPT=1  DRY_RUN=1  FG=1
# Run file: $OUT/<game>_<algo><suffix><_TAG>_s<seed>.jsonl  (Stoix json log; last line {"done": true})
# =====================================================================================================================
CWQ_LAUNCHER=$(readlink -f "$0"); source "$(dirname "$CWQ_LAUNCHER")/lib/common.sh"; cd "$STOIX_DIR"
BENCH=minatar
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0   # autotune segfaults otherwise
ALGOS=${ALGOS:-"cwq dqn ddqn dueling c51 rc51 rc"}
GAMES=${GAMES:-"asterix breakout freeway seaquest space_invaders"}
# use common seeds for all algorithms to reduce variance in comparisons
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
NW_PER_GPU=${NW_PER_GPU:-3}
OUT=${OUT:-$RUNS_ROOT/minatar_main}
RHO_DEFAULT=0.5; G_DEFAULT=8; RHO=${RHO:-$RHO_DEFAULT}; G=${G:-$G_DEFAULT}; NONORM=${NONORM:-1}; NSTEP=${NSTEP:-1}; RHO_CROSS_K=${RHO_CROSS_K:-0}
TT=${TT:-5e6}; NUM_EVAL=${NUM_EVAL:-50}; EPOCHS=${EPOCHS:-8}; LR=${LR:-5e-4}; BATCH=${BATCH:-512}; CKPT=${CKPT:-1}
TAG=${TAG:-}
[ "$TT$NUM_EVAL$EPOCHS$LR$BATCH" != "5e65085e-4512" ] && cwq_require_tag_if "TT/NUM_EVAL/EPOCHS/LR/BATCH"
[ -n "${EXTRA_ALL:-}${EXTRA_OURS:-}" ] && cwq_require_tag_if "EXTRA_ALL/EXTRA_OURS"
CODE_SNAPSHOT=(stoix/systems/q_learning/ff_rdqn.py stoix/systems/q_learning/ff_rpair.py stoix/systems/q_learning/same_traj.py stoix/systems/q_learning/ff_pqn_m.py)

C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=$TT arch.num_evaluation=$NUM_EVAL system.run_log_dir=$OUT"
P="$C system.rollout_length=1 system.epochs=$EPOCHS system.total_batch_size=$BATCH system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=$LR system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
[ "$CKPT" = 1 ] && P="$P +system.ckpt_dir=$OUT/ckpt"
RDQN="stoix/systems/q_learning/ff_rdqn.py $P system.rdqn=False system.c_lr=0.0"
PAIR="stoix/systems/q_learning/ff_rpair.py $P system.term=cross +system.all_pairs=True +system.pair_rank=0 +system.duel_cur_target=True system.same_traj=True system.st_group=$G system.pair_rho=$RHO"
[ "$NONORM" = 1 ] && PAIR="$PAIR +system.cw_no_norm=True" || PAIR="$PAIR +system.cw_no_norm=False"
[ "$NSTEP" != 1 ] && PAIR="$PAIR system.n_step=$NSTEP"
[ "$RHO_CROSS_K" != 0 ] && PAIR="$PAIR +system.rho_cross_k=$RHO_CROSS_K"
SUF=$(cwq_ours_suffix); TSUF=${TAG:+_$TAG}

is_ours() { case $1 in cwq|cwq_duel) return 0 ;; *) return 1 ;; esac; }
run_name() { if is_ours $1; then echo "$1$SUF$TSUF"; else echo "$1$TSUF"; fi; }
algo_cmd() {
  case $1 in
    cwq)      echo "$PAIR system.head=qcw ${EXTRA_OURS:-}" ;;
    cwq_duel) echo "$PAIR system.head=duel ${EXTRA_OURS:-}" ;;
    dqn)      echo "$RDQN network=minatar_dqn" ;;
    ddqn)     echo "$RDQN network=minatar_dqn system.double=True" ;;
    dueling)  echo "$RDQN network=minatar_dueling_dqn" ;;
    c51)      echo "$RDQN network=minatar_dqn system.c51=True system.num_atoms=51 system.vmin=-20 system.vmax=20" ;;
    rc)       echo "$RDQN network=minatar_dqn system.reward_centering=True system.rc_beta=0.001" ;;
    rc51)     echo "stoix/systems/q_learning/ff_rpair.py $P system.head=rc51 system.term=unif system.cont_coef=0.33 +system.rc_atoms=51 +system.rc_vmax=20" ;;
    pqn)      echo "stoix/systems/q_learning/ff_pqn_m.py $C network=minatar_dqn_ln system.rollout_length=32 system.epochs=2 system.num_minibatches=8 system.q_lr=$LR system.max_grad_norm=10 system.q_lambda=0.65 system.gamma=0.99 system.training_epsilon=0.1 system.decay_epsilon=True system.eps_decay_steps=100000" ;;
    *) echo "unknown algorithm $1" >&2; exit 1 ;;
  esac
}
job_cmd() { echo "$PY $(algo_cmd $2) ${EXTRA_ALL:-} env=pgx/$1 arch.seed=$3 system.algo_name=$(run_name $2)"; }
job_out() { echo "$OUT/${1}_$(run_name $2)_s$3.jsonl"; }
job_env() { :; }
for a in $ALGOS; do algo_cmd $a > /dev/null || exit 1; done
JOBS=(); for s in $SEEDS; do for g in $GAMES; do for a in $ALGOS; do JOBS+=("$g $a $s"); done; done; done
cwq_main "$@"
