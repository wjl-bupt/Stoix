"""2026-11-10 (user): qcw all-pairs -- build the whole stop-grad target sg(D + xi/(1-g)) from the TARGET net
(D^- replaces the online D in the W4 term and in the outer D); only the regressed -D_ij stays online.
Flag +system.qcw_full_target=True (needs duel_cur_target=True). Default off = bit-identical."""
import shutil
F = "/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py"
shutil.copy(F, F + ".bak_20261110_before_fulltgt")
s = open(F).read()
a = "                   cf_same: bool = False, cf_mixed: bool = True, n_step: int = 1):"
assert s.count(a) == 1; s = s.replace(a, a[:-2] + ", full_target: bool = False):")
i0 = s.index("    def duel_all_pairs("); i1 = s.index("    def mlp_all_pairs(")
f = s[i0:i1]
b = "        Dsg = jax.lax.stop_gradient(Dm)\n        xi = "
assert f.count(b) == 1
f = f.replace(b, "        Dsg = jax.lax.stop_gradient(Dm)\n"
                 "        if full_target:                                                 # 2026-11-10: whole target from the target net\n"
                 "            assert cur_target, \"qcw_full_target needs duel_cur_target=True\"\n"
                 "            Dsg = jax.lax.stop_gradient(wmat(c_sg, c_sg) + A_sg[:, None] - A_sg[None, :])\n"
                 "        xi = ")
c = "jax.lax.stop_gradient(Dm + delta) - Dm"
assert f.count(c) == 2; f = f.replace(c, "jax.lax.stop_gradient(Dsg + delta) - Dm")
s = s[:i0] + f + s[i1:]
d = 'cur_target=bool(config.system.get("duel_cur_target", False)),'
assert s.count(d) == 2
s = s.replace(d, d + '\n                                full_target=bool(config.system.get("qcw_full_target", False)),')
open(F, "w").write(s); print("patched")
