cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/smoke_new; mkdir -p $OUT
C="arch.total_num_envs=128 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=1e6 arch.num_evaluation=5 arch.seed=0 system.run_log_dir=$OUT env=pgx/asterix env.wrapper=null"
P="$C system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000"
ulimit -s unlimited
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
run() { local name=$1; shift; for k in 1 2 3; do CUDA_VISIBLE_DEVICES=6 .venv/bin/python "$@" > $OUT/$name.log 2>&1; grep -q done $OUT/asterix_${name}_s0.jsonl 2>/dev/null && break; done; }
run cwpair_lr stoix/systems/q_learning/ff_rpair.py $P system.algo_name=cwpair_lr system.head=lowrank system.term=cross +system.all_pairs=True +system.pair_rank=32 &
run cwpair_mlp stoix/systems/q_learning/ff_rpair.py $P system.algo_name=cwpair_mlp system.head=free system.term=cross +system.n_partners=8 &
run rc51 stoix/systems/q_learning/ff_rpair.py $P system.algo_name=rc51 system.head=rc51 system.term=unif &
run pqn stoix/systems/q_learning/ff_pqn_m.py $C network=minatar_dqn_ln system.algo_name=pqn system.rollout_length=32 system.epochs=2 system.num_minibatches=8 system.q_lr=5e-4 system.max_grad_norm=10 system.q_lambda=0.65 system.gamma=0.99 system.training_epsilon=0.1 system.decay_epsilon=True system.eps_decay_steps=100000 &
wait
