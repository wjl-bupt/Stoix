cd /home/junweiluo/Stoix
OUT=/home/junweiluo/stoix_runs/det_test; rm -rf $OUT; mkdir -p $OUT
P="env.wrapper=null network=minatar_dqn arch.total_num_envs=128 system.rollout_length=1 system.epochs=8 system.total_batch_size=512 system.warmup_steps=40 system.total_buffer_size=100000 system.q_lr=5e-4 system.max_grad_norm=10 system.eps_decay_steps=100000 system.target_period=1000 system.evaluation_epsilon=0.0 arch.absolute_metric=False arch.total_timesteps=4e5 arch.num_evaluation=2 arch.seed=0 system.run_log_dir=$OUT env=pgx/asterix"
ulimit -s unlimited
export XLA_PYTHON_CLIENT_PREALLOCATE=false TF_CPP_MIN_LOG_LEVEL=3 XLA_FLAGS=--xla_gpu_autotune_level=0
for k in 1 2 3; do
  for a in new_a new_b; do
    [ -f $OUT/asterix_${a}_s0.jsonl ] && grep -q done $OUT/asterix_${a}_s0.jsonl && continue
    CUDA_VISIBLE_DEVICES=7 .venv/bin/python stoix/systems/q_learning/ff_rdqn.py $P system.algo_name=$a > $OUT/$a.log 2>&1
  done
done
cp /tmp/ff_rdqn_old.py stoix/systems/q_learning/ff_rdqn_old.py
sed -i "s#^class MinAtarTorso#class _UnusedTorso#" stoix/systems/q_learning/ff_rdqn_old.py
for k in 1 2 3; do
  [ -f $OUT/asterix_old_s0.jsonl ] && grep -q done $OUT/asterix_old_s0.jsonl && break
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python stoix/systems/q_learning/ff_rdqn_old.py $P system.algo_name=old > $OUT/old.log 2>&1
done
rm -f stoix/systems/q_learning/ff_rdqn_old.py
