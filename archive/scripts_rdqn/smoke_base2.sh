cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/smoke_base2; mkdir -p $OUT
P="env.wrapper=null network=minatar_dqn arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=1e6 arch.num_evaluation=5 arch.seed=0 system.run_log_dir=$OUT env=pgx/asterix"
ulimit -s unlimited
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
CUDA_VISIBLE_DEVICES=6 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=dqn > $OUT/dqn.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=c51 system.c51=True > $OUT/c51.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=rc system.reward_centering=True > $OUT/rc.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=ddqn system.double=True > $OUT/ddqn.log 2>&1 &
wait
