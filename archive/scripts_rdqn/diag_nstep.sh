cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/diag_nstep; mkdir -p $OUT
P="env.wrapper=null network=minatar_dqn arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=1e6 arch.num_evaluation=10 env=gymnax/freeway arch.seed=0 system.run_log_dir=$OUT"
ulimit -s unlimited
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
CUDA_VISIBLE_DEVICES=3 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=dqn2 system.n_step=2 > $OUT/dqn2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=dqn3_eps01 system.n_step=3 system.eps_end=0.01 > $OUT/dqn3e.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=dqn3_polyak system.n_step=3 system.target_period=0 system.tau=0.005 > $OUT/dqn3p.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=dqn3_lr1e4 system.n_step=3 system.q_lr=1e-4 > $OUT/dqn3lr.log 2>&1 &
wait
