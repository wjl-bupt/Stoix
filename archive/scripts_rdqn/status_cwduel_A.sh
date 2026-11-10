#!/bin/bash
# progress of CW-Duel stage A: last logged timestep / eval_return / sps of every run
OUT=/home/junweiluo/stoix_runs/cwduel_A
for f in $OUT/*_cwduel_*.jsonl; do
  [ -f "$f" ] || continue
  /home/junweiluo/Stoix/.venv/bin/python - "$f" <<'EOF'
import json, sys, os
L = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs = [r for r in L if "timestep" in r]; done = any(r.get("done") for r in L)
if recs:
    r = recs[-1]
    print(f"{os.path.basename(sys.argv[1]):34s} t={r['timestep']/1e6:4.2f}M eval={r['eval_return']:6.2f} "
          f"sps={r['sps']:7.0f} adv_range={r.get('adv_range', float('nan')):.3f} {'DONE' if done else ''}")
else:
    print(os.path.basename(sys.argv[1]), "no records yet")
EOF
done
tail -3 $OUT/_logs/progress.txt 2>/dev/null
ls $OUT | grep -E "ALLDONE|STOPPED"
