#!/bin/bash
# alpha=0 with al_mode=al must reproduce al_mode=none exactly (CPU, deterministic); alpha=0.9 must log al_pen>0.
cd /home/junweiluo/Stoix; O=/tmp/alpal_wiring; rm -rf $O; mkdir -p $O
C="env.wrapper=null arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=3e4 arch.num_evaluation=2 system.run_log_dir=$O"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 env=pgx/asterix arch.seed=0"
R="stoix/systems/q_learning/ff_rdqn.py $P network=minatar_dqn system.rdqn=True system.c_lr=1.0 system.same_traj=True system.st_group=8"
W="stoix/systems/q_learning/ff_rpair.py $P system.head=free system.term=cross +system.n_partners=8 system.same_traj=True system.st_group=8 system.st_cross=7"
run() { JAX_PLATFORMS=cpu .venv/bin/python "$@" > $O/$(echo "$@" | md5sum | cut -c1-8).log 2>&1; }
run $R system.algo_name=r_none &
run $R system.algo_name=r_a0 system.al_mode=al system.al_alpha=0.0 &
run $R system.algo_name=r_p9 system.al_mode=pal system.al_alpha=0.9 &
run $W system.algo_name=w_none &
run $W system.algo_name=w_a0 system.al_mode=pal system.al_alpha=0.0 &
run $W system.algo_name=w_a9 system.al_mode=al system.al_alpha=0.9 &
wait; ls $O
