#!/bin/bash
B=/home/junweiluo/gpu_busy; kill -9 $(cat $B/mgr.pid) 2>/dev/null
for f in $B/holder_*.pid; do kill -9 $(cat $f) 2>/dev/null; rm -f $f; done; echo stopped
