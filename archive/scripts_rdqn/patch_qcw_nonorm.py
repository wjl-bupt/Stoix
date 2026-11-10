"""2026-11-10 (user): qcw/duel all-pairs -- drop the 1/(1-gamma) normalisation of xi (delta = xi instead of xi/(1-gamma)).
Flag +system.cw_no_norm=True. Default off = bit-identical.  (Pure global scale x(1-gamma) of loss gradient; only Adam eps / clip see it.)"""
import shutil
F = "/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py"
shutil.copy(F, F + ".bak_20261110_before_nonorm")
s = open(F).read()
a = ", full_target: bool = False):"
assert s.count(a) == 1; s = s.replace(a, ", full_target: bool = False, no_norm: bool = False):")
i0 = s.index("    def duel_all_pairs("); i1 = s.index("    def mlp_all_pairs(")
f = s[i0:i1]
b = "        vx, vy = 1 - Gx, 1 - Gy\n        W1 = k_ * Gx * Gy"
assert f.count(b) == 1
f = f.replace(b, "        if no_norm:                                                     # 2026-11-10: no 1/(1-gamma) on xi\n"
                 "            nrm_ = 1.0\n" + b)
s = s[:i0] + f + s[i1:]
d = 'full_target=bool(config.system.get("qcw_full_target", False)),'
assert s.count(d) == 2
s = s.replace(d, d + '\n                                no_norm=bool(config.system.get("cw_no_norm", False)),')
open(F, "w").write(s); print("patched")
