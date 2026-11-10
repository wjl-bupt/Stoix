#!/bin/bash
# =====================================================================================================================
# DeepMind Control (MuJoCo Playground / MJX) launcher: CW-Q-rho family + baselines.   bash cwq/run_dmc.sh
#
# Protocol (= MuJoCo Playground official brax_sac_config for DM Control, with the deviations we fixed on 11-04/11-06):
#   128 envs, rollout 1, 8 gradient updates per vector step (UTD 1/16), batch 512, replay 4M (2^22), lr 1e-3 actor+critic
#   (Adam eps 1e-5), gamma .99, tau .005, Q-net LayerNorm, MLP 256x256 ReLU (actor and critic), no grad clip,
#   64 warm-up vector steps of uniform random actions, truncation bootstrapped, NO observation normalisation,
#   TD3 mechanics (expl noise .1, target noise .2 clip .5, policy delay 2); DDPG = one critic, no smoothing, actor every step;
#   SAC = Brax-style auto alpha.  5M env steps (10M for acrobot/finger/hopper/cheetah/humanoid/walker_run), seeds 0-9,
#   50 evaluations x 32 episodes.  Note: DMC has no terminations -> cross weighting is inactive; CW-Q-rho here =
#   batch-centred TD + rho pair kernel.
#
# Algorithms (ALGOS):
#   ours       cwq_td3     CW-Q-rho on TD3 with the level-free (batch-centred) clipped double-Q min   [old arm cwq_cmin]
#              cwq_td3raw  CW-Q-rho on TD3 with the raw min (hijacked by level drift, see README)      [old arm cwq_rho]
#              cwq_ddpg    CW-Q-rho on DDPG                                                        [old arm cwq_ddpg]
#   baselines  td3 ddpg sac
#   default    ALGOS="cwq_td3 cwq_ddpg td3 ddpg sac"
# Knobs of OUR algorithms: RHO=0.5 G=8 NONORM=1 (0 = old xi/(1-gamma); name suffix _norm)   EXTRA_OURS (needs TAG)
# Protocol knobs: STEPS (override 5M/10M table for all tasks; needs TAG)  EXTRA_ALL (needs TAG)
# Selection / resources:
#   TASKS="walker_walk walker_run hopper_hop finger_spin reacher_hard cheetah_run"  SEEDS="0 ... 9"
#   GPU_LIST="0 1 2"  NW_PER_GPU=1 (MJX segfaults when jobs share a GPU)  MINFREE=10000 (MB; each run needs ~8.6 GB)
#   RETRIES=30  OUT=$RUNS_ROOT/dmc_main  DRY_RUN=1  FG=1
# Run file: $OUT/<TaskName>_<algo><suffix><_TAG>_s<seed>.jsonl     (TaskName from the env yaml, e.g. WalkerWalk)
# =====================================================================================================================
CWQ_LAUNCHER=$(readlink -f "$0"); source "$(dirname "$CWQ_LAUNCHER")/lib/common.sh"; cd "$STOIX_DIR"
BENCH=dmc
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1 JAX_COMPILATION_CACHE_DIR=/home/junweiluo/.jax_cache_dmc JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_command_buffer= --xla_gpu_enable_llvm_module_compilation_parallelism=false"   # driver segfault workarounds
ALGOS=${ALGOS:-"cwq_td3 cwq_ddpg td3 ddpg sac"}
TASKS=${TASKS:-"walker_walk walker_run hopper_hop finger_spin reacher_hard cheetah_run"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
NW_PER_GPU=${NW_PER_GPU:-1}; MINFREE=${MINFREE:-10000}
OUT=${OUT:-$RUNS_ROOT/dmc_main}
RHO_DEFAULT=0.5; G_DEFAULT=8; RHO=${RHO:-$RHO_DEFAULT}; G=${G:-$G_DEFAULT}; NONORM=${NONORM:-1}
TAG=${TAG:-}; STEPS=${STEPS:-}
[ -n "$STEPS" ] && cwq_require_tag_if "STEPS"
[ -n "${EXTRA_ALL:-}${EXTRA_OURS:-}" ] && cwq_require_tag_if "EXTRA_ALL/EXTRA_OURS"
CODE_SNAPSHOT=(stoix/systems/ddpg/cw_continuous.py stoix/systems/ddpg/ff_cw_td3.py stoix/systems/ddpg/ff_cw_ddpg.py stoix/systems/ddpg/ff_cw_sac.py)

DMC_C="arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.total_buffer_size=4194304 \
system.actor_lr=1e-3 system.q_lr=1e-3 system.decay_learning_rates=False system.max_grad_norm=1e9 system.tau=0.005 system.gamma=0.99 \
system.exploration_noise=0.1 system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 \
system.warmup_random=True system.warmup_steps=64 system.q_hidden=[256,256] system.activation=relu system.layer_norm=True \
network.actor_network.pre_torso.layer_sizes=[256,256] network.actor_network.pre_torso.activation=relu \
system.truncation_bootstrap=True arch.absolute_metric=False arch.num_evaluation=50 arch.num_eval_episodes=32"
DDPG_C=${DMC_C/system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 /}
OURS="system.head=qcw system.pairing=rho system.pair_rho=$RHO system.st_group=$G"
[ "$NONORM" = 1 ] && OURS="$OURS +system.no_norm=True" || OURS="$OURS +system.no_norm=False"
SUF=$(cwq_ours_suffix); TSUF=${TAG:+_$TAG}

steps_of() { [ -n "$STEPS" ] && { echo $STEPS; return; }
  case $1 in acrobot_swingup|finger_spin|hopper_hop|hopper_stand|cheetah_run|humanoid_walk|walker_run) echo 1e7 ;; *) echo 5e6 ;; esac; }
is_ours() { case $1 in cwq_td3|cwq_td3raw|cwq_ddpg) return 0 ;; *) return 1 ;; esac; }
run_name() { if is_ours $1; then echo "$1$SUF$TSUF"; else echo "$1$TSUF"; fi; }
algo_cmd() {
  case $1 in
    cwq_td3)    echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C $OURS +system.min_mode=centered ${EXTRA_OURS:-}" ;;
    cwq_td3raw) echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C $OURS ${EXTRA_OURS:-}" ;;
    cwq_ddpg)   echo "stoix/systems/ddpg/ff_cw_ddpg.py $DDPG_C $OURS ${EXTRA_OURS:-}" ;;
    td3)        echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C system.head=q" ;;
    ddpg)       echo "stoix/systems/ddpg/ff_cw_ddpg.py $DDPG_C system.head=q" ;;
    sac)        echo "stoix/systems/ddpg/ff_cw_sac.py $DMC_C system.head=q arch.evaluation_greedy=True" ;;
    *) echo "unknown algorithm $1" >&2; exit 1 ;;
  esac
}
declare -A TNAME
for t in $TASKS; do TNAME[$t]=$($PY -c "import yaml,sys; print(yaml.safe_load(open('stoix/configs/env/mjc_playground/dm_control/$t.yaml'))['scenario']['task_name'])") || exit 1; done
job_cmd() { local ex=""; [ $1 = cheetah_run ] && ex="+env.batch_cond_reset=True +system.no_batch_vmap=True"   # 85x faster resets, see README
  echo "timeout 36000 $PY $(algo_cmd $2) ${EXTRA_ALL:-} env=mjc_playground/dm_control/$1 $ex arch.total_timesteps=$(steps_of $1) arch.seed=$3 system.algo_name=$(run_name $2) system.run_log_dir=$OUT"; }
job_out() { echo "$OUT/${TNAME[$1]}_$(run_name $2)_s$3.jsonl"; }
job_env() { :; }
for a in $ALGOS; do algo_cmd $a > /dev/null || exit 1; done
JOBS=(); for s in $SEEDS; do for t in $TASKS; do for a in $ALGOS; do JOBS+=("$t $a $s"); done; done; done
cwq_main "$@"
