import json, sys, glob, os
out = sys.argv[1] if len(sys.argv) > 1 else "/home/junweiluo/stoix_runs/same_traj"
for f in sorted(glob.glob(os.path.join(out, "*.jsonl"))):
    L = [json.loads(x) for x in open(f) if x.strip()]
    r = [x for x in L if "timestep" in x]
    if not r: print(os.path.basename(f), "no records"); continue
    x = r[-1]
    print(f"{os.path.basename(f):34s} t={x['timestep']:>8d} eval={x.get('eval_return', 0):7.2f} chain={x.get('st_chain_ok', -1):.3f} "
          f"pairfrac={x.get('st_pair_frac', -1):.3f} len={x.get('st_traj_len', -1):7.1f} |dt|={x.get('st_abs_dt', -1):6.1f} "
          f"sps={int(x['sps'])} {'DONE' if 'done' in L[-1] else ''}")
