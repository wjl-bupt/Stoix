#!/bin/bash
cd /home/junweiluo/Stoix
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
timeout 900 .venv/bin/python scripts_rdqn/test_crn_step.py > scripts_rdqn/test_crn_step.log 2>&1
echo EXIT $? >> scripts_rdqn/test_crn_step.log
