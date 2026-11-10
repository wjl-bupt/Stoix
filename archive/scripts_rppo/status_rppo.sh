#!/bin/bash
OUT=/home/junweiluo/stoix_runs/minatar_rppo_10m
echo "running: $(pgrep -fc 'ff_rppo.py.*minatar_rppo_10m')  done: $(grep -l '"done": true' $OUT/*.jsonl 2>/dev/null | wc -l)/20"
for f in $OUT/*.jsonl; do /home/junweiluo/Stoix/.venv/bin/python - "$f" <<'PY'
import json,sys,os
L=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
r=[x for x in L if "timestep" in x]
if r: x=r[-1]; print(f"{os.path.basename(sys.argv[1]):28s} t={x['timestep']/1e6:5.2f}M eval={x.get('eval_return',0):7.2f} c={x.get('c_end',0):7.3f} vhat={x.get('v_hat_rollout_mean',0):7.3f} rm={x.get('resid_mean',0):+.4f} rs={x.get('resid_std',0):.3f} sps={x['sps']:.0f}")
PY
done
