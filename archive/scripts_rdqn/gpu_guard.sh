#!/bin/bash
# Run an experiment launcher on GPU0/1 with the placeholder released, and put the placeholder back afterwards
# (also when the launcher is stopped / killed).  usage (detached):
#   nohup setsid bash scripts_rdqn/gpu_guard.sh bash scripts_rdqn/launch_xxx.sh > guard.out 2>&1 < /dev/null &
cd /home/junweiluo/Stoix
bash scripts_rdqn/placeholder.sh stop
trap 'bash /home/junweiluo/Stoix/scripts_rdqn/placeholder.sh start' EXIT
"$@"
