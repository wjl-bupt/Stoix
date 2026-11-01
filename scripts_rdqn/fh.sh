#!/bin/bash
cd /home/junweiluo/Stoix
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1 TF_CPP_MIN_LOG_LEVEL=3
timeout 600 .venv/bin/python -X faulthandler stoix/systems/q_learning/ff_dqn.py env=gymnax/asterix arch.total_timesteps=3e5 arch.num_evaluation=3 arch.seed=7 arch.absolute_metric=False > scripts_rdqn/fh.log 2>&1
echo EXIT $? >> scripts_rdqn/fh.log
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 >> scripts_rdqn/fh.log
.venv/bin/python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)" >> scripts_rdqn/fh.log 2>&1
.venv/bin/pip list 2>/dev/null | grep -i -E "nvidia-cudnn|nvidia-cublas|jax" >> scripts_rdqn/fh.log
