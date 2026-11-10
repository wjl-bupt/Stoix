cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/smoke_rpair; mkdir -p $OUT
P="env.wrapper=null arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=1e6 arch.num_evaluation=10 env=gymnax/asterix arch.seed=0 system.run_log_dir=$OUT"
ulimit -s unlimited
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
i=0
for h in anti free; do for t in cross unif; do
CUDA_VISIBLE_DEVICES=$i .venv/bin/python stoix/systems/q_learning/ff_rpair.py $P system.head=$h system.term=$t system.algo_name=pair_${h}_${t} > $OUT/${h}_${t}.log 2>&1 &
i=$((i+1)); done; done
wait
