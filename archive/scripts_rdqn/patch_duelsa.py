"""2026-11-04: add head=duel_sa to ff_rpair.py (CW-Duel with a pairwise (s,a) advantage head).
run from /home/junweiluo/Stoix:  .venv/bin/python scripts_rdqn/patch_duelsa.py"""
import shutil, os
P = "stoix/systems/q_learning/ff_rpair.py"
BAK = P + ".bak_20261104_before_duelsa"
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
s = open(BAK).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:60], n, count)
    s = s.replace(old, new)


rep("Old single-partner description:",
    "head=duel_sa (2026-11-04, CW-Duel with a pairwise (s,a) advantage head): identical to head=duel (same W, same loss,\n"
    "  all pairs, rho weights, cur_target) except the advantage: psi(s,a) = (s,a) encoder e(s,a) of the free/anti heads\n"
    "  (Dense128(relu(Dense128([phi(s), emb(a)]))), 16-d action embedding), psi_c(s,a) = psi(s,a) - mean_b psi(s,b)\n"
    "  (centring in embedding space, every action passed through the encoder once), A(s,a) = w . psi_c(s,a) with a\n"
    "  LINEAR read-out w (no bias).  Hence A(s,a) - A(t,b) = w . (psi_c(s,a) - psi_c(t,b)), rows of A sum to 0, and the\n"
    "  same-state pairwise matrix delta_ab = w . (psi(s,a) - psi(s,b)) has row mean A(s,a) (= Borda).  A non-linear\n"
    "  read-out of the difference would not decompose and would reintroduce the free-head non-identifiability.\n"
    "Old single-partner description:")
rep("        self.du_w = nn.Dense(1)                   # duel: w(s)\n",
    "        self.du_w = nn.Dense(1)                   # duel: w(s)\n"
    "        self.du_wa = nn.Dense(1, use_bias=False)  # duel_sa: linear read-out w of the centred (s,a) embedding\n")
rep("        adv = self.du_adv(h)\n        Ac = adv - adv.mean(-1, keepdims=True)\n",
    "        if self.head == \"duel_sa\":     # 2026-11-04: A(s,a) = w . (psi(s,a) - mean_b psi(s,b))\n"
    "            Ac = self.duel_sa_adv(h)\n"
    "        else:\n"
    "            adv = self.du_adv(h)\n"
    "            Ac = adv - adv.mean(-1, keepdims=True)\n")
rep("    def duel_comp_obs(self, obs):\n",
    "    def sa_all(self, h):\n"
    "        \"\"\"(s,a) encodings psi(s, b) for every action b: state features h [..., F] -> [..., A, E].\"\"\"\n"
    "        nA = self.num_actions\n"
    "        a = jnp.broadcast_to(jnp.arange(nA), h.shape[:-1] + (nA,))\n"
    "        hb = jnp.broadcast_to(h[..., None, :], h.shape[:-1] + (nA, h.shape[-1]))\n"
    "        return self.sa(hb, a)\n\n"
    "    def duel_sa_adv(self, h):\n"
    "        \"\"\"duel_sa centred advantage [..., A]: w . (psi(s,a) - mean_b psi(s,b)).\"\"\"\n"
    "        psi = self.sa_all(h)\n"
    "        psi_c = psi - psi.mean(-2, keepdims=True)\n"
    "        return self.du_wa(psi_c)[..., 0]\n\n"
    "    def duel_comp_obs(self, obs):\n")
rep('        if self.head == "duel":', '        if self.head in ("duel", "duel_sa"):', count=2)
rep('        if net.head == "duel":', '        if net.head in ("duel", "duel_sa"):')
rep('or (str(config.system.head) == "duel" and', 'or (str(config.system.head) in ("duel", "duel_sa") and')
rep('                if str(config.system.head) == "duel":', '                if str(config.system.head) in ("duel", "duel_sa"):')
rep('"rc51", "duel", "duel_mlp")', '"rc51", "duel", "duel_mlp", "duel_sa")')
open(P, "w").write(s)
print("patched", P)
