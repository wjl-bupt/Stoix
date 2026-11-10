"""Checks for system.nonterm_only (2026-11-02).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_nonterm.py
N1 flag off: loss / grads identical to the code path before the change (mpl default).
N2 flag on: loss = 0.5 * mean over pairs with d_x=d_y=0 of delta^2 (manual, from return_delta).
N3 flag on: changing reward / next_obs of terminal transitions leaves loss and gradient unchanged.
N4 flag on with no terminals in the batch: identical to flag off.
N5 nn_pair_frac logged and equal to the fraction of double non-terminal pairs.
N6 forbidden combos raise (all_pairs, lowrank, crn).
"""
import sys
sys.path.insert(0, "/home/junweiluo/Stoix")
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss as mpl
from stoix.systems.q_learning.dqn_types import Transition
ok = tot = 0
def chk(n, c, e=""):
    global ok, tot; tot += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + n + " " + e)
rng = np.random.default_rng(0); A, g, m = 5, 0.99, 64
obs = jnp.array(rng.integers(0, 2, size=(m, 10, 10, 4)), jnp.float32)
nxt = jnp.array(rng.integers(0, 2, size=(m, 10, 10, 4)), jnp.float32)
done = jnp.array(rng.random(m) < .15)
tr = Transition(obs=obs, action=jnp.array(rng.integers(0, A, m)), reward=jnp.array(rng.normal(size=m), jnp.float32),
                done=done, next_obs=nxt, info={})
net = PairNet(num_actions=A, head="free"); p = net.init(jax.random.PRNGKey(1), obs[:1])
tp = jax.tree_util.tree_map(lambda x: x * 1.01, p)
sh = jnp.int32(7)
l0, c0 = mpl(net, g, "cross", .33, 1e9, n_partners=8, return_delta=True)
lF, _ = mpl(net, g, "cross", .33, 1e9, n_partners=8, return_delta=True, nonterm_only=False)
lT, _ = mpl(net, g, "cross", .33, 1e9, n_partners=8, return_delta=True, nonterm_only=True)
leaves = lambda t: jax.tree_util.tree_leaves(t)
gd = lambda a, b: max(float(jnp.abs(x - y).max()) for x, y in zip(leaves(a), leaves(b)))
G = lambda f, t: jax.grad(lambda q: f(q, tp, t, sh)[0])(p)
chk("N1 flag off == default loss", float(l0(p, tp, tr, sh)[0]) == float(lF(p, tp, tr, sh)[0]))
chk("N1 flag off == default grad", gd(G(l0, tr), G(lF, tr)) == 0.0)
# manual masked loss
K = 8; shifts = 1 + (7 - 1 + jnp.arange(K) * ((m - 1) // K)) % (m - 1)
ix = jnp.tile(jnp.arange(m), K); iy = ((jnp.arange(m)[None, :] + shifts[:, None]) % m).reshape(-1)
d = np.asarray(done, np.float32); nn = (1 - d[np.asarray(ix)]) * (1 - d[np.asarray(iy)])
lossT, infoT = lT(p, tp, tr, sh)
delta = np.asarray(l0(p, tp, tr, sh)[1]["_delta"])
want = 0.5 * np.sum(nn * delta ** 2) / nn.sum()
chk("N2 masked loss = manual", abs(float(lossT) - want) < 1e-5 * max(1, want), f"{float(lossT):.6f} vs {want:.6f} (all-pair loss {float(l0(p,tp,tr,sh)[0]):.4f})")
# N3: perturb terminal transitions' reward and next_obs
dm = np.asarray(done)
r2 = np.asarray(tr.reward).copy(); r2[dm] += 37.0
n2 = np.asarray(nxt).copy(); n2[dm] = 1 - n2[dm]
tr2 = tr._replace(reward=jnp.array(r2), next_obs=jnp.array(n2))
chk("N3 terminal transitions do not affect loss", abs(float(lT(p, tp, tr2, sh)[0]) - float(lossT)) < 1e-6)
chk("N3 terminal transitions do not affect grad", gd(G(lT, tr), G(lT, tr2)) < 1e-7, f"{gd(G(lT, tr), G(lT, tr2)):.1e}")
chk("N3 (control) flag off IS affected", abs(float(l0(p, tp, tr2, sh)[0]) - float(l0(p, tp, tr, sh)[0])) > 1e-3)
tr0 = tr._replace(done=jnp.zeros(m, bool))
chk("N4 no terminals: on == off", abs(float(lT(p, tp, tr0, sh)[0]) - float(l0(p, tp, tr0, sh)[0])) < 1e-6 and gd(G(lT, tr0), G(l0, tr0)) < 1e-7)
chk("N5 nn_pair_frac", abs(float(infoT["nn_pair_frac"]) - nn.mean()) < 1e-6, f"{float(infoT['nn_pair_frac']):.3f}")
for kw, nt in ((dict(all_pairs=True), net), (dict(), PairNet(num_actions=A, head="lowrank", rank=4)), (dict(use_cf=True), net)):
    try:
        mpl(nt, g, "cross", .33, 1e9, nonterm_only=True, **kw); chk(f"N6 raises {kw}", False)
    except AssertionError:
        chk(f"N6 raises {kw} head={nt.head}", True)
print(f"{ok}/{tot} passed")
