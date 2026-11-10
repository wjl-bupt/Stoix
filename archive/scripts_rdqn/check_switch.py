"""check_switch (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_switch.py"""
import importlib.machinery, importlib.util
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
from stoix.systems.q_learning.same_traj import rho_weight_matrix
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss, duel_dqn_loss
from stoix.systems.q_learning.dqn_types import Transition
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
rng = np.random.default_rng(0); m = 48; A = 3; g = 0.99
obs = jnp.array(rng.random((m, 10, 10, 4)) < 0.2, jnp.float32); nxt = jnp.roll(obs, 1, 0)
tr = Transition(obs=obs, action=jnp.array(rng.integers(0, A, m)), reward=jnp.array(rng.normal(size=m), jnp.float32),
                done=jnp.array(rng.random(m) < 0.15), next_obs=nxt, info={})
net = PairNet(num_actions=A, head="duel", rank=0)
p = net.init(jax.random.PRNGKey(0), obs[:1]); tp = net.init(jax.random.PRNGKey(1), obs[:1])
w, _, _, Ac = net.apply(p, obs, method=PairNet.duel_comp_obs); wn, _, _, Acn = net.apply(tp, nxt, method=PairNet.duel_comp_obs)
Q = w[:, None] + Ac; Qn = wn[:, None] + Acn
d = tr.done.astype(jnp.float32)
y = tr.reward + g * (1 - d) * Qn.max(-1)
ref = 0.5 * jnp.mean(jnp.square(y - Q[jnp.arange(m), tr.action]))
l = duel_dqn_loss(net, p, tp, tr, g, 1000.0)
check("duel_dqn_loss = 0.5 mean (r + g(1-d) max Q-(s') - Q(s,a))^2, Q = w + A", abs(float(l) - float(ref)) < 1e-5 * max(1, float(ref)), (float(l), float(ref)))
gr = jax.grad(lambda q: duel_dqn_loss(net, q, tp, tr, g, 1000.0))(p)["params"]
check("dqn loss: grads reach torso, du_w, du_adv; target params get none",
      all(float(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(gr[k]))) > 0 for k in ["torso", "du_w", "du_adv"]))
gt = jax.grad(lambda q: duel_dqn_loss(net, p, q, tr, g, 1000.0))(tp)
check("no gradient through the target", all(float(jnp.max(jnp.abs(x))) == 0 for x in jax.tree_util.tree_leaves(gt)))
# CW path unchanged vs pre-switch file
bak = "stoix/systems/q_learning/ff_rpair.py.bak_20261104_before_switch"
ldr = importlib.machinery.SourceFileLoader("ff_rpair_old", bak)
spec = importlib.util.spec_from_loader("ff_rpair_old", ldr); old = importlib.util.module_from_spec(spec); ldr.exec_module(old)
no = old.PairNet(num_actions=A, head="duel", rank=0)
W = rho_weight_matrix(jnp.array(rng.integers(0, 30, m)), 8, 0.5); ar = jnp.arange(m)
_, cn = make_pair_loss(net, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
_, co = old.make_pair_loss(no, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
check("CW loss unchanged vs pre-patch", float(cn(p, tp, tr, ar, ar, W)[0]) == float(co(p, tp, tr, ar, ar, W)[0]))
print(f"{sum(ok)}/{len(ok)} passed")
