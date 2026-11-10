# 2026-11-08 Kinetix off-policy protocol (user: 1e8 env steps, transformer for all methods; PPO later).
# Official Kinetix generalist protocol: train on randomly generated S levels (DR), evaluate zero-shot on the 10
# hand-designed S holdout levels (solve rate, 10 episodes per level).  Continuous actions (6 dims in [-1,1]),
# symbolic-entity observations, gamma .995, frame skip 2, dense reward scale .2 (Kinetix defaults).
# Network: official-size transformer encoder (enc 128, qkv 16, 8 heads, 2 layers) + 5x128 MLP, separate for actor/critics.
# Off-policy: 512 envs, 4 updates per vector step (UTD 1/128), batch 512, replay 1M, Adam 3e-4, grad clip 0.5 (Kinetix),
# tau .005, 32k uniform-random warm-up transitions, TD3 noise .1 / smoothing .2,.5 / delay 2.  Eval every 1M steps.
E=${KIN_E:-512}; EP=${KIN_EPOCHS:-4}
KIN_C="arch.total_num_envs=$E system.rollout_length=1 system.epochs=$EP system.total_batch_size=512 system.total_buffer_size=1048576 \
system.actor_lr=3e-4 system.q_lr=3e-4 system.decay_learning_rates=False system.max_grad_norm=0.5 system.tau=0.005 system.gamma=0.995 \
system.exploration_noise=0.1 system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 \
system.warmup_random=True system.warmup_steps=64 system.q_hidden=[128,128,128,128,128] system.activation=relu system.layer_norm=False \
+system.obs_encoder=kinetix_tf system.truncation_bootstrap=True network=kinetix_tf_ddpg \
env=kinetix/small env/kinetix/train=random env.scenario.action_type=continuous \
arch.absolute_metric=False arch.num_evaluation=${KIN_NEVAL:-100} arch.num_eval_episodes=100"
kin_cmd() {  # $1 arm
  case $1 in
    td3)      echo "stoix/systems/ddpg/ff_cw_td3.py $KIN_C system.head=q" ;;
    cwq_cmin) echo "stoix/systems/ddpg/ff_cw_td3.py $KIN_C system.head=qcw system.pairing=rho system.pair_rho=0.5 system.st_group=8 +system.min_mode=centered" ;;
    ddpg)     echo "stoix/systems/ddpg/ff_cw_ddpg.py ${KIN_C/system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 /} system.head=q" ;;
    cwq_ddpg) echo "stoix/systems/ddpg/ff_cw_ddpg.py ${KIN_C/system.policy_noise=0.2 system.noise_clip=0.5 system.policy_frequency=2 /} system.head=qcw system.pairing=rho system.pair_rho=0.5 system.st_group=8" ;;
    sac)      echo "stoix/systems/ddpg/ff_cw_sac.py $KIN_C system.head=q arch.evaluation_greedy=True" ;;
  esac
}
