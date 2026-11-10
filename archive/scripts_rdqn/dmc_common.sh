# 2026-11-04 DMC (MuJoCo Playground) community-aligned TD3 protocol (Fujimoto 2018 / CleanRL td3_continuous_action):
# UTD 1 (epochs = num_envs, rollout 1), batch 256, buffer 1M, Adam 3e-4 (actor+critic), no lr decay, no grad clip,
# tau .005, gamma .99, expl noise .1, policy noise .2 / clip .5, policy delay 2, 25k uniform-random warmup steps,
# actor/critic MLP 256x256 ReLU, truncation bootstrapped (done = true termination), 1M env steps, eval every 10k.
E=${E:-16}
DMC_C="arch.total_num_envs=$E system.rollout_length=1 system.epochs=$E system.total_batch_size=256 system.total_buffer_size=1000000 \
system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=False system.max_grad_norm=1e9 system.tau=0.005 system.gamma=0.99 \
system.exploration_noise=0.1 system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 \
system.warmup_random=True system.warmup_steps=$((25000 / E)) system.q_hidden=[256,256] system.activation=relu \
network.actor_network.pre_torso.layer_sizes=[256,256] network.actor_network.pre_torso.activation=relu \
system.truncation_bootstrap=True arch.absolute_metric=False arch.num_eval_episodes=16"
dmc_cmd() {  # $1 arm
  case $1 in
    td3)      echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C system.head=q" ;;
    cwq_rho)  echo "stoix/systems/ddpg/ff_cw_td3.py $DMC_C system.head=qcw system.pairing=rho system.pair_rho=0.5 system.st_group=8" ;;
  esac
}
