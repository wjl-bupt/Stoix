"""Checks for strat_term / term_partners (2026-11-02).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_vr.py
V1 no terminals in the batch (pi_T=0): pw = 1/n -> weighted loss / grad == original loss_fn (K partners).
V2 strat_sample: exactly n_T terminal items when the buffer has terminals, pi_T exact, invalid slots never sampled;
   buffer without terminals -> pi_T = 0 and all items valid non-terminal.
V3 unbiasedness (Monte Carlo, scalar pair function f on a fake buffer): E[sum pw f] = E_{x,y iid} f for
   plain / post-strat / strat / multi / strat+multi; reports bias in SE units and the variance ratio vs plain.
V4 weights: sum pw = 1 when all cells present; pairs of a cell share pi_h pi_h' equally.
"""
import sys
sys.path.insert(0, "/home/junweiluo/Stoix")
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss, strat_sample, build_weighted_pairs, build_rowcol_pairs
from stoix.systems.q_learning.dqn_types import Transition
ok = tot = 0
def chk(n, c, e=""):
    global ok, tot; tot += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + n + " " + e)
rng = np.random.default_rng(0); A, g, m, K = 5, 0.99, 64, 8
obs = jnp.array(rng.integers(0, 2, size=(m, 10, 10, 4)), jnp.float32)
tr = Transition(obs=obs, action=jnp.array(rng.integers(0, A, m)), reward=jnp.array(rng.normal(size=m), jnp.float32),
                done=jnp.zeros(m, bool), next_obs=obs[::-1], info={})
net = PairNet(num_actions=A, head="free"); p = net.init(jax.random.PRNGKey(1), obs[:1])
loss_fn, compute = make_pair_loss(net, g, "cross", .33, 1e9, n_partners=K)
sh = jnp.int32(9)
ix, iy, pw = build_weighted_pairs(jax.random.PRNGKey(0), tr.done, jnp.float32(0.0), K, sh, 0, 32)
l0 = loss_fn(p, p, tr, sh)[0]; lw = compute(p, p, tr, ix, iy, pw=pw)[0]
g0 = jax.grad(lambda q: loss_fn(q, p, tr, sh)[0])(p); gw = jax.grad(lambda q: compute(q, p, tr, ix, iy, pw=pw)[0])(p)
gd = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(gw)))
chk("V1 pi_T=0, no extras: weighted == original loss", abs(float(l0 - lw)) < 1e-5 * max(1, abs(float(l0))), f"{float(l0):.6f} {float(lw):.6f}")
chk("V1 grad", gd < 1e-6, f"{gd:.1e}")
# V2
L = 300
dd = rng.random(L) < 0.03; valid = np.arange(L) < 250
exp = Transition(obs=jnp.arange(L), action=jnp.zeros(L, int), reward=jnp.zeros(L), done=jnp.array(dd), next_obs=jnp.arange(L), info={})
b, piT = strat_sample(exp, jnp.array(valid), jax.random.PRNGKey(3), 64, 16)
ids = np.asarray(b.obs)
chk("V2 exactly n_T terminals", int(np.asarray(b.done).sum()) == 16 and dd[ids].sum() == 16)
chk("V2 pi_T exact", abs(float(piT) - (dd & valid).sum() / valid.sum()) < 1e-7, f"{float(piT):.4f}")
chk("V2 only valid slots", valid[ids].all())
exp0 = exp._replace(done=jnp.zeros(L, bool))
b0, piT0 = strat_sample(exp0, jnp.array(valid), jax.random.PRNGKey(4), 64, 16)
chk("V2 no terminals: pi_T=0, valid items", float(piT0) == 0.0 and valid[np.asarray(b0.obs)].all())
# V3 Monte Carlo unbiasedness on a fake buffer with scalar items
L = 400; mm = 128; KK = 4
d = (rng.random(L) < 0.02); d[:3] = True
val = rng.normal(size=L) + 6 * d                      # terminal items carry large values (like amplified deltas)
F = lambda a, b: a * 1.0 + 0.5 * b + 2.0 * a * b      # pair function
truth = float(np.mean(F(val[:, None], val[None, :])))
expS = Transition(obs=jnp.arange(L), action=jnp.zeros(L, int), reward=jnp.zeros(L), done=jnp.array(d), next_obs=jnp.arange(L), info={})
vJ = jnp.array(val); validS = jnp.ones(L, bool); piTS = jnp.float32(d.mean())
def est(key, mode):
    k1, k2, k3 = jax.random.split(key, 3)
    sh = jax.random.randint(k3, (), 1, mm)
    if mode in ("strat", "strat_multi"):
        bt, pi = strat_sample(expS, validS, k1, mm, 16)
    else:
        idx = jax.random.randint(k1, (mm,), 0, L)
        bt = jax.tree_util.tree_map(lambda x: x[idx], expS); pi = piTS
    x = vJ[bt.obs]
    if mode == "plain":
        sh_ = 1 + (sh - 1 + jnp.arange(KK) * ((mm - 1) // KK)) % (mm - 1)
        ix = jnp.tile(jnp.arange(mm), KK); iy = ((jnp.arange(mm)[None, :] + sh_[:, None]) % mm).reshape(-1)
        return jnp.mean(F(x[ix], x[iy]))
    KT = 16 if mode in ("multi", "strat_multi") else 0
    if mode == "post_cells_BIASED":
        ix, iy, pw = build_weighted_pairs(k2, bt.done, pi, KK, sh, 0, 16)
    elif mode == "multi":
        ix, iy, pw = build_rowcol_pairs(k2, bt.done, KK, sh, KT, 16)
    else:
        ix, iy, pw = build_weighted_pairs(k2, bt.done, pi, KK, sh, KT, 16, tt=True)
    return jnp.sum(pw * F(x[ix], x[iy]))
NREP = 40000
res = {}
for mode in ("plain", "strat", "multi", "strat_multi", "post_cells_BIASED"):
    e = np.asarray(jax.jit(jax.vmap(lambda k: est(k, mode)))(jax.random.split(jax.random.PRNGKey(11), NREP)))
    se = e.std() / np.sqrt(NREP); res[mode] = e.var()
    z = (e.mean() - truth) / se
    if mode == "post_cells_BIASED":   # documents why (B) uses row/column weights: cell weights + uniform batch = empty cells
        print(f"INFO V3 {mode}: mean {e.mean():.4f} truth {truth:.4f} z={z:+.2f}"); continue
    chk(f"V3 {mode}: unbiased", abs(z) < 4.0, f"mean {e.mean():.4f} truth {truth:.4f} z={z:+.2f}  var/var_plain={e.var()/res['plain']:.3f}")
# V4
dd4 = jnp.array((np.arange(64) % 9) == 0)
ix, iy, pw = build_weighted_pairs(jax.random.PRNGKey(5), dd4, jnp.float32(0.1), 8, jnp.int32(5), 8, 16, tt=True)
chk("V4 weights sum to 1", abs(float(pw.sum()) - 1) < 1e-5, f"{float(pw.sum()):.6f}")
hx, hy = np.asarray(dd4)[np.asarray(ix)], np.asarray(dd4)[np.asarray(iy)]
pwn = np.asarray(pw)
for a, b_ in ((0, 0), (0, 1), (1, 0), (1, 1)):
    c = (hx == a) & (hy == b_) & (pwn > 0)
    w = (0.9 if a == 0 else 0.1) * (0.9 if b_ == 0 else 0.1)
    chk(f"V4 cell ({a},{b_}) total weight {w:.3f}", abs(pwn[c].sum() - w) < 1e-5 and np.ptp(pwn[c]) < 1e-7)
# V5 rowcol: KT=0 equals plain mean weights; weights sum to 1 with extras; no terminals -> plain
ix, iy, pw = build_rowcol_pairs(jax.random.PRNGKey(6), jnp.zeros(64, bool), 8, jnp.int32(5), 0, 16)
chk("V5 rowcol KT=0 = 1/(mK)", float(jnp.abs(pw - 1 / (64 * 8)).max()) < 1e-9)
ix, iy, pw = build_rowcol_pairs(jax.random.PRNGKey(6), dd4, 8, jnp.int32(5), 56, 16)
chk("V5 rowcol weights sum to 1", abs(float(pw.sum()) - 1) < 1e-5, f"{float(pw.sum()):.6f}")
ix, iy, pw = build_rowcol_pairs(jax.random.PRNGKey(6), jnp.zeros(64, bool), 8, jnp.int32(5), 56, 16)
chk("V5 rowcol no terminals: extras get weight 0", float(pw[64 * 8:].sum()) == 0.0 and abs(float(pw[:64 * 8].sum()) - 1) < 1e-6)
print(f"{ok}/{tot} passed")
