"""check_rho_duel (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_rho_duel.py"""
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.same_traj import rho_weight_matrix
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.dqn_types import Transition
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
rng = np.random.default_rng(0)
G, ng = 8, 6; m = G * ng
idx = rng.integers(0, 30, m)
W = np.asarray(rho_weight_matrix(jnp.array(idx), G, 0.5))
grp = np.arange(m) // G
same = grp[:, None] == grp[None, :]; dist = idx[:, None] != idx[None, :]
check("W: zero on self / coincident items", np.all(W[~dist] == 0))
check("W: same-group part sums 0.5, cross part sums 0.5", abs(W[same].sum() - .5) < 1e-6 and abs(W[~same].sum() - .5) < 1e-6)
check("W: uniform inside each part", np.allclose(W[same & dist], W[same & dist][0]) and np.allclose(W[~same & dist], W[~same & dist][0]))
# duel all-pairs loss: pmat = mask / (m(m-1)) must reproduce the unweighted loss exactly
A = 5; obs = jnp.array(rng.random((m, 10, 10, 4)) < 0.2, jnp.float32)
net = PairNet(num_actions=A, head="duel", rank=32)
p = net.init(jax.random.PRNGKey(0), obs[:1]); tp = net.init(jax.random.PRNGKey(1), obs[:1])
tr = Transition(obs=obs, action=jnp.array(rng.integers(0, A, m)), reward=jnp.array(rng.normal(size=m), jnp.float32),
                done=jnp.array(rng.random(m) < 0.1), next_obs=jnp.roll(obs, 1, 0), info={})
_, compute = make_pair_loss(net, 0.99, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
l0, _ = compute(p, tp, tr, jnp.arange(m), jnp.arange(m), None)
pm = (1 - jnp.eye(m)) / (m * (m - 1))
l1, _ = compute(p, tp, tr, jnp.arange(m), jnp.arange(m), pm)
check("duel loss with uniform pmat == unweighted loss", abs(float(l0) - float(l1)) < 1e-6 * max(1, abs(float(l0))), (float(l0), float(l1)))
g0 = jax.grad(lambda q: compute(q, tp, tr, jnp.arange(m), jnp.arange(m), None)[0])(p)
g1 = jax.grad(lambda q: compute(q, tp, tr, jnp.arange(m), jnp.arange(m), pm)[0])(p)
err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)))
check("duel gradient with uniform pmat == unweighted gradient", err < 1e-6, err)
l2, _ = compute(p, tp, tr, jnp.arange(m), jnp.arange(m), jnp.array(W))
check("rho-weighted duel loss finite and different", np.isfinite(float(l2)) and abs(float(l2) - float(l0)) > 0)
print(f"{sum(ok)}/{len(ok)} passed")
