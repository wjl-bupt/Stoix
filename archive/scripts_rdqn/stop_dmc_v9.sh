#!/bin/bash
# stop launcher v9 (workers + running jobs) and release our GPU holders
for p in $(pgrep -f "bash scripts_rdqn/launch_dm[c].sh"); do kill -9 $p; done
for p in $(pgrep -f "ff_cw_(td3|ddpg|sac).p[y]"); do kill -9 $p; done
for p in $(pgrep -f "scripts_rdqn/gpu_hold.p[y]"); do kill -9 $p; done
echo stopped
