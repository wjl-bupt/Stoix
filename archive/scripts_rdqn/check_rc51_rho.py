import sys; sys.path.insert(0, "/home/junweiluo/Stoix")
from types import SimpleNamespace
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.same_traj import rho_pairs
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
A, m, G = 3, 64, 8
net = PairNet(num_actions=A, head="rc51", hidden=512, emb_dim=48, rank=32, n_atoms=51, vmax=20.0, additive=False)
rng = np.random.default_rng(0)
obs = jnp.asarray(rng.random((m, 10, 10, 4)) < .2).astype(jnp.float32)
p = net.init(jax.random.PRNGKey(0), obs); tp = jax.tree_util.tree_map(lambda x: x * 1.1, p)
tr = SimpleNamespace(obs=obs, next_obs=jnp.asarray(rng.random((m, 10, 10, 4)) < .2).astype(jnp.float32),
                     action=jnp.asarray(rng.integers(0, A, m)), reward=jnp.asarray((rng.random(m) < .2).astype(np.float32)),
                     done=jnp.asarray(rng.random(m) < .1))
_, comp = make_pair_loss(net, 0.99, "unif", 0.33, 1000.0)
ix0 = jnp.arange(m); iy0 = jnp.roll(ix0, 5)
l_none, _ = comp(p, tp, tr, ix0, iy0)
l_unif, _ = comp(p, tp, tr, ix0, iy0, pw=jnp.full(m, 1.0 / m))
chk(abs(float(l_none - l_unif)) < 1e-5, f"uniform pw == plain mean ({float(l_none):.5f} vs {float(l_unif):.5f})")
h = m // 2
l_half, _ = comp(p, tp, tr, ix0[:h], iy0[:h])
pw_h = jnp.concatenate([jnp.full(h, 1.0 / h), jnp.zeros(m - h)])
l_wh, _ = comp(p, tp, tr, ix0, iy0, pw=pw_h)
chk(abs(float(l_half - l_wh)) < 1e-5, f"weights select pairs ({float(l_half):.5f} vs {float(l_wh):.5f})")
# rho pairs structure on a group batch
idx = jnp.asarray(rng.integers(0, 10**6, m)); grp = np.arange(m) // G
ixs = jnp.asarray(np.concatenate([[g * G + a for a in range(G) for b in range(G) if a != b] for g in range(m // G)]))
iys = jnp.asarray(np.concatenate([[g * G + b for a in range(G) for b in range(G) if a != b] for g in range(m // G)]))
ws = jnp.full(ixs.shape[0], 1.0 / ixs.shape[0])
ix, iy, pw = rho_pairs(jax.random.PRNGKey(1), idx, m // G, G, 7, 0.5, ixs, iys, ws)
same = grp[np.asarray(ix)] == grp[np.asarray(iy)]
chk(abs(float(pw[same].sum()) - .5) < 1e-5 and abs(float(pw[~same].sum()) - .5) < 1e-5, "rho pairs: same 1-rho, cross rho")
g_, _ = jax.grad(lambda pp: comp(pp, tp, tr, ix, iy, pw=pw), has_aux=True)(p)
chk(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(g_)), "gradient finite with rho pairs")
print(f"{ok}/{n} passed")
