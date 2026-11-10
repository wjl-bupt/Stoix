#!/bin/bash
H=/home/junweiluo/stoix_runs/mujoco_gym/_hold_mj
for f in $H/pid_*; do kill -9 $(cat $f) 2>/dev/null; rm -f $f; done; rm -f $H/held_*
