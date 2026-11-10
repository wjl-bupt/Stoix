# 2026-11-08: optional parameter checkpoints at 20/40/60/80/100% of training (for same-state action-ranking tests).
# Off unless system.ckpt_dir (Stoix) / --ckpt_dir (mujoco) is given -> training is bit-identical otherwise.
import shutil
R = "/home/junweiluo/Stoix/"
def patch(f, edits, tag="bak_20261108_before_ckpt"):
    s = open(R + f).read()
    for old, new in edits:
        assert s.count(old) == 1, (f, old[:60], s.count(old))
        s = s.replace(old, new)
    shutil.copy(R + f, R + f + "." + tag); open(R + f, "w").write(s); print("patched", f)
anchor = '        flog.write(json.dumps(rec) + "\\n"); flog.flush()\n'
hook = anchor + (
 '        _ck = config.system.get("ckpt_dir")                # 2026-11-08 checkpoints at 20/40/60/80/100%\n'
 '        if _ck and (eval_step + 1) % max(1, config.arch.num_evaluation // 5) == 0:\n'
 '            import pickle\n'
 '            os.makedirs(_ck, exist_ok=True)\n'
 '            with open(os.path.join(_ck, f"{config.env.scenario.task_name}_{algo}_s{config.arch.seed}_t{t}.pkl"), "wb") as _fh:\n'
 '                pickle.dump(jax.device_get(trained_params), _fh)\n')
for f in ["stoix/systems/q_learning/ff_rdqn.py", "stoix/systems/q_learning/ff_rpair.py"]:
    patch(f, [(anchor, hook)])
m = "scripts_mujoco/mujoco_offpolicy.py"
patch(m, [
 ('    ap.add_argument("--out", required=True)\n',
  '    ap.add_argument("--out", required=True)\n'
  '    ap.add_argument("--ckpt_dir", default="")             # 2026-11-08: save actor/critic at 20/40/60/80/100%\n'),
 ('            flog.write(json.dumps(rec) + "\\n"); flog.flush()\n',
  '            flog.write(json.dumps(rec) + "\\n"); flog.flush()\n'
  '        if A.ckpt_dir and t % (A.total // 5) == 0:\n'
  '            import pickle\n'
  '            os.makedirs(A.ckpt_dir, exist_ok=True)\n'
  '            _nm = os.path.basename(A.out).replace(".jsonl", "")\n'
  '            with open(os.path.join(A.ckpt_dir, f"{_nm}_t{t}.pkl"), "wb") as _fh:\n'
  '                pickle.dump(jax.device_get({"actor": ap_, "q": qp, "args": vars(A)}), _fh)\n'),
])
import ast
for f in ["stoix/systems/q_learning/ff_rdqn.py", "stoix/systems/q_learning/ff_rpair.py", m]:
    ast.parse(open(R + f).read())
print("syntax ok")
