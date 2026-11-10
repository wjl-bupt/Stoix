"""check_qcw (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_qcw.py
head=qcw: D = Q(s,a) - Q(t,b), plain DQN head, cross-weighted pair loss, no scalar c."""
import importlib.machinery, importlib.util
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
from stoix.systems.q_learning.same_traj import rho_weight_matrix
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.dqn_types import Transition

ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
rng = np.random.default_rng(0)
G, ng = 8, 6; m = G * ng; A = 5; g = 0.99
obs = jnp.array(rng.random((m, 10, 10, 4)) < 0.2, jnp.float32)
nxt = jnp.roll(obs, 1, 0)
act = jnp.array(rng.integers(0, A, m)); rew = jnp.array(rng.normal(size=m), jnp.float32)
done = jnp.array(rng.random(m) < 0.15)
net = PairNet(num_actions=A, head="qcw", rank=0)
p = net.init(jax.random.PRNGKey(0), obs[:1]); tp = net.init(jax.random.PRNGKey(1), obs[:1])
check("params = torso + du_adv only", set(p["params"].keys()) == {"torso", "du_adv"}, sorted(p["params"].keys()))
Qf = lambda q, o: net.apply(q, o, method=PairNet.duel_comp_obs)[3]
Q = Qf(p, obs)
iy = jnp.roll(jnp.arange(m), 5)
D = net.apply(p, obs, act, obs[iy], act[iy], method=PairNet.duel_D)
check("D = Q(s,a) - Q(t,b)", float(jnp.max(jnp.abs(D - (Q[jnp.arange(m), act] - Q[iy, act[iy]])))) < 1e-5)
check("greedy = argmax Q", bool(jnp.all(jnp.argmax(net.apply(p, obs, method=PairNet.scores), -1) == jnp.argmax(Q, -1))))
tr = Transition(obs=obs, action=act, reward=rew, done=done, next_obs=nxt, info={})
ar = jnp.arange(m)
# xi with target = online equals v_j u_i - v_i u_j, u = DQN TD error (no c)
_, comp = make_pair_loss(net, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True, return_delta=True)
_, info = comp(p, p, tr, ar, ar, None)
d = done.astype(jnp.float32); v = 1 - g * (1 - d)
Qn = Qf(p, nxt)
u = rew + g * (1 - d) * Qn.max(-1) - Q[ar, act]
xi_ref = (v[None, :] * u[:, None] - v[:, None] * u[None, :]) / (1 - g)
mask = 1 - jnp.eye(m)
err = float(jnp.max(jnp.abs((info["_delta"] - xi_ref) * mask)))
check("delta_ij = (v_j u_i - v_i u_j)/(1-g) when target = online", err < 1e-3 * max(1.0, float(jnp.max(jnp.abs(xi_ref)))), err)
# zero set: if u = kappa * v for every item, all pair residuals vanish
kap = 0.7
rew2 = Q[ar, act] - g * (1 - d) * Qn.max(-1) + kap * v           # makes u = kappa v exactly
_, info2 = comp(p, p, Transition(obs=obs, action=act, reward=rew2, done=done, next_obs=nxt, info={}), ar, ar, None)
check("u = kappa (1-g(1-d)) -> all pair residuals 0", float(jnp.max(jnp.abs(info2["_delta"] * mask))) < 1e-2, float(jnp.max(jnp.abs(info2["_delta"] * mask))))
rew3 = Q[ar, act] - g * (1 - d) * Qn.max(-1) + kap                # u = kappa (RDQN-without-c zero set)
_, info3 = comp(p, p, Transition(obs=obs, action=act, reward=rew3, done=done, next_obs=nxt, info={}), ar, ar, None)
check("u = kappa (const) is NOT a zero of the cross-weighted residual when some items terminate",
      float(jnp.max(jnp.abs(info3["_delta"] * mask))) > 1.0, float(jnp.max(jnp.abs(info3["_delta"] * mask))))
# loss weighting and gradients
_, comp2 = make_pair_loss(net, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
l0 = comp2(p, tp, tr, ar, ar, None)[0]; l1 = comp2(p, tp, tr, ar, ar, (1 - jnp.eye(m)) / (m * (m - 1)))[0]
check("uniform pmat == unweighted", abs(float(l0) - float(l1)) < 1e-6 * max(1, abs(float(l0))), (float(l0), float(l1)))
idx = jnp.array(rng.integers(0, 30, m)); W = rho_weight_matrix(idx, G, 0.5)
gr = jax.grad(lambda q: comp2(q, tp, tr, ar, ar, W)[0])(p)["params"]
check("gradient reaches torso and du_adv", all(float(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(gr[k]))) > 0 for k in ["torso", "du_adv"]))
# regression: head=duel (rank 0 and 32) unchanged vs pre-patch file
bak = "stoix/systems/q_learning/ff_rpair.py.bak_20261104_before_qcw"
ldr = importlib.machinery.SourceFileLoader("ff_rpair_old", bak)
spec = importlib.util.spec_from_loader("ff_rpair_old", ldr); old = importlib.util.module_from_spec(spec); ldr.exec_module(old)
for r in (0, 32):
    nd = PairNet(num_actions=A, head="duel", rank=r); no = old.PairNet(num_actions=A, head="duel", rank=r)
    pd = nd.init(jax.random.PRNGKey(2), obs[:1]); tpd = nd.init(jax.random.PRNGKey(3), obs[:1])
    _, cn = make_pair_loss(nd, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
    _, co = old.make_pair_loss(no, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
    gn = jax.grad(lambda q: cn(q, tpd, tr, ar, ar, W)[0])(pd); go = jax.grad(lambda q: co(q, tpd, tr, ar, ar, W)[0])(pd)
    e = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(gn), jax.tree_util.tree_leaves(go)))
    check(f"head=duel rank {r}: loss & grad identical to pre-patch", float(cn(pd, tpd, tr, ar, ar, W)[0]) == float(co(pd, tpd, tr, ar, ar, W)[0]) and e == 0.0, e)
print(f"params qcw {sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(p))}")
print(f"{sum(ok)}/{len(ok)} passed")
