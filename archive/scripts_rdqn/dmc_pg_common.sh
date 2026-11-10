# 2026-11-04 DMC main protocol = MuJoCo Playground official Brax SAC config for DM Control Suite
# (mujoco_playground/config/dm_control_suite_params.py brax_sac_config), with a TD3 base (Stoix ff_td3 mechanics):
#   128 envs, rollout 1, 8 grad updates per vector step (UTD 1/16), batch 512, lr 1e-3 (actor+critic), gamma .99,
#   tau .005, replay 4M (2^22), min replay 8192 (warmup 64 vector steps, uniform random actions), Q-net LayerNorm,
#   MLP 256x256 ReLU, no grad clip, no lr decay, action repeat 1, truncation bootstrapped.
#   5M env steps; 10M for Acrobot/Finger/Hopper/CheetahRun/HumanoidWalk/WalkerRun (as Playground).
#   11-06: arms ddpg (one critic, no smoothing, actor every step) and sac (Brax-style, auto alpha) added.
#   Deviations from Playground: TD3 instead of SAC (expl noise .1, target noise .2/.5, policy delay 2);
#   NO observation normalisation (user rule: off-policy runs without normalisation wrappers); 50 evals x 32 episodes.
E=128
DMC_C="arch.total_num_envs=$E system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.total_buffer_size=4194304 \
system.actor_lr=1e-3 system.q_lr=1e-3 system.decay_learning_rates=False system.max_grad_norm=1e9 system.tau=0.005 system.gamma=0.99 \
system.exploration_noise=0.1 system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 \
system.warmup_random=True system.warmup_steps=64 system.q_hidden=[256,256] system.activation=relu system.layer_norm=True \
network.actor_network.pre_torso.layer_sizes=[256,256] network.actor_network.pre_torso.activation=relu \
system.truncation_bootstrap=True arch.absolute_metric=False arch.num_evaluation=50 arch.num_eval_episodes=32"
dmc_steps() { case $1 in acrobot_swingup|finger_spin|hopper_hop|hopper_stand|cheetah_run|humanoid_walk|walker_run) echo 1e7 ;; *) echo 5e6 ;; esac; }
dmc_cmd() {  # $1 arm
  case $1 in
    td3)      echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C system.head=q" ;;
    cwq_rho)  echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C system.head=qcw system.pairing=rho system.pair_rho=0.5 system.st_group=8" ;;
    cwq_cmin) echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C system.head=qcw system.pairing=rho system.pair_rho=0.5 system.st_group=8 +system.min_mode=centered" ;;
    ddpg)     echo "stoix/systems/ddpg/ff_cw_ddpg.py ${DMC_C/system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 /} system.head=q" ;;
    cwq_ddpg) echo "stoix/systems/ddpg/ff_cw_ddpg.py ${DMC_C/system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 /} system.head=qcw system.pairing=rho system.pair_rho=0.5 system.st_group=8" ;;
    sac)      echo "stoix/systems/ddpg/ff_cw_sac.py $DMC_C system.head=q arch.evaluation_greedy=True" ;;
  esac
}
