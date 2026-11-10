#!/bin/bash
for p in $(pgrep -f "launch_dmc_v1[0].sh"); do kill -9 $p; done
for p in $(pgrep -f "ff_cw_(td3|ddpg|sac).p[y]"); do kill -9 $p; done
rm -f /home/junweiluo/gpu_busy/busy_*; echo stopped   # holder manager re-holds the GPUs
