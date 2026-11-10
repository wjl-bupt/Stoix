"""2026-11-04: add head=qcw to ff_rpair.py: cross-weighted pair loss on a plain DQN head (no scalar c).
D((s,a),(t,b)) = Q(s,a) - Q(t,b), Q(s,.) = Dense_A(phi(s)) (same torso as head=duel, no Dueling split, no W).
Everything else = head=duel path (all pairs, rho weight matrix, cross weighting, cur_target).
run from /home/junweiluo/Stoix:  .venv/bin/python scripts_rdqn/patch_qcw.py"""
import shutil, os
P = "stoix/systems/q_learning/ff_rpair.py"
BAK = P + ".bak_20261104_before_qcw"
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
s = open(BAK).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:70], n, count)
    s = s.replace(old, new)


rep("Old single-partner description:",
    "head=qcw (2026-11-04, 'RDQN without c, cross-weighted'): D((s,a),(t,b)) = Q(s,a) - Q(t,b) with a plain DQN head\n"
    "  Q(s,.) = Dense_A(phi(s)) (du_adv, NOT centred; w = 0, no state-pair term).  Loss/pairs/targets = head=duel path.\n"
    "  Zero set of the cross-weighted residual: u = kappa (1 - g(1-d)) -> Q = Q* - kappa (level only, policy-invariant),\n"
    "  unlike the centred RDQN loss without c (u = kappa -> Q*_{r-kappa}).\n"
    "Old single-partner description:")
rep("        if self.head == \"duel_sa\":     # 2026-11-04: A(s,a) = w . (psi(s,a) - mean_b psi(s,b))\n"
    "            Ac = self.duel_sa_adv(h)\n",
    "        if self.head == \"qcw\":         # 2026-11-04: plain Q head, no centring, no w / W\n"
    "            Ac = self.du_adv(h)\n"
    "            z = jnp.zeros(h.shape[:-1] + (0,), h.dtype)\n"
    "            return jnp.zeros(h.shape[:-1], h.dtype), z, z, Ac\n"
    "        if self.head == \"duel_sa\":     # 2026-11-04: A(s,a) = w . (psi(s,a) - mean_b psi(s,b))\n"
    "            Ac = self.duel_sa_adv(h)\n")
rep('        if self.head in ("duel", "duel_sa"):', '        if self.head in ("duel", "duel_sa", "qcw"):', count=2)
rep('        if net.head in ("duel", "duel_sa"):', '        if net.head in ("duel", "duel_sa", "qcw"):')
rep('or (str(config.system.head) in ("duel", "duel_sa") and', 'or (str(config.system.head) in ("duel", "duel_sa", "qcw") and')
rep('                if str(config.system.head) in ("duel", "duel_sa"):', '                if str(config.system.head) in ("duel", "duel_sa", "qcw"):')
rep('"rc51", "duel", "duel_mlp", "duel_sa")', '"rc51", "duel", "duel_mlp", "duel_sa", "qcw")')
open(P, "w").write(s)
print("patched", P)
