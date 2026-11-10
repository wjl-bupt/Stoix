"""check_duel_sa (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_duel_sa.py
head=duel_sa: A(s,a) = w . (psi(s,a) - mean_b psi(s,b)), D = W(s,t) + A(s,a) - A(t,b)."""
import importlib.machinery, importlib.util
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
from stoix.systems.q_learning.same_traj import rho_weight_matrix
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.dqn_types import Transition

ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
rng = np.random.default_rng(0)
G, ng = 8, 6; m = G * ng; A = 5
obs = jnp.array(rng.random((m, 10, 10, 4)) < 0.2, jnp.float32)
act = jnp.array(rng.integers(0, A, m))
net = PairNet(num_actions=A, head="duel_sa", rank=32)
p = net.init(jax.random.PRNGKey(0), obs[:1]); tp = net.init(jax.random.PRNGKey(1), obs[:1])
names = set(p["params"].keys())
check("params: encoder + linear w present, Dueling du_adv absent",
      {"torso", "aemb", "sa1", "sa2", "du_wa", "du_w", "du_z1", "du_z2"} <= names and "du_adv" not in names, sorted(names))
check("w has no bias", set(p["params"]["du_wa"].keys()) == {"kernel"})

w, z1, z2, Ac = net.apply(p, obs, method=PairNet.duel_comp_obs)
check("A rows sum to 0", float(jnp.max(jnp.abs(Ac.sum(-1)))) < 1e-5, float(jnp.max(jnp.abs(Ac.sum(-1)))))

# independent computation: psi(s,b) one action at a time through PairNet.sa
h = net.apply(p, obs, method=PairNet.phi)
psi = jnp.stack([net.apply(p, h, jnp.full((m,), b), method=PairNet.sa) for b in range(A)], 1)   # [m, A, E]
wv = p["params"]["du_wa"]["kernel"][:, 0]
f = psi @ wv                                                                        # f(s,b) = w . psi(s,b)
check("A = f - mean_b f (per-action encoder passes)", float(jnp.max(jnp.abs(Ac - (f - f.mean(1, keepdims=True))))) < 1e-5)
delta = jnp.einsum("sabe,e->sab", psi[:, :, None, :] - psi[:, None, :, :], wv)      # same-state w.(psi_a - psi_b)
check("same-state pairwise matrix antisymmetric", float(jnp.max(jnp.abs(delta + jnp.swapaxes(delta, 1, 2)))) < 1e-5)
check("A = row mean of same-state pairwise matrix", float(jnp.max(jnp.abs(delta.mean(2) - Ac))) < 1e-5)
sc = net.apply(p, obs, method=PairNet.scores)
off = 1 - jnp.eye(A)
check("scores = Borda (mean over b != a)", float(jnp.max(jnp.abs(sc - (delta * off).sum(2) / (A - 1)))) < 1e-5)
check("greedy = argmax A", bool(jnp.all(jnp.argmax(sc, -1) == jnp.argmax(Ac, -1))))

# cross-state pairs: D = W + w.(psi_c(s,a) - psi_c(t,b))
iy = jnp.roll(jnp.arange(m), 3)
D = net.apply(p, obs, act, obs[iy], act[iy], method=PairNet.duel_D)
psic = psi - psi.mean(1, keepdims=True)
Wst = w - w[iy] + jnp.sum(z1 * z2[iy], -1) - jnp.sum(z2 * z1[iy], -1)
Dm = Wst + (psic[jnp.arange(m), act] - psic[iy, act[iy]]) @ wv
check("cross-state D = W + w.(psi_c(s,a) - psi_c(t,b))", float(jnp.max(jnp.abs(D - Dm))) < 1e-5)
D2 = net.apply(p, obs[iy], act[iy], obs, act, method=PairNet.duel_D)
check("D antisymmetric, D(x,x) = 0", float(jnp.max(jnp.abs(D + D2))) < 1e-5
      and float(jnp.max(jnp.abs(net.apply(p, obs, act, obs, act, method=PairNet.duel_D)))) < 1e-6)

# loss: uniform pmat == unweighted; rho weights finite; gradients reach encoder and w
tr = Transition(obs=obs, action=act, reward=jnp.array(rng.normal(size=m), jnp.float32),
                done=jnp.array(rng.random(m) < 0.1), next_obs=jnp.roll(obs, 1, 0), info={})
_, compute = make_pair_loss(net, 0.99, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
ar = jnp.arange(m)
l0, info = compute(p, tp, tr, ar, ar, None)
pm = (1 - jnp.eye(m)) / (m * (m - 1))
l1, _ = compute(p, tp, tr, ar, ar, pm)
check("loss with uniform pmat == unweighted", abs(float(l0) - float(l1)) < 1e-6 * max(1, abs(float(l0))), (float(l0), float(l1)))
idx = jnp.array(rng.integers(0, 30, m))
l2, _ = compute(p, tp, tr, ar, ar, rho_weight_matrix(idx, G, 0.5))
check("rho-weighted loss finite", np.isfinite(float(l2)))
g = jax.grad(lambda q: compute(q, tp, tr, ar, ar, rho_weight_matrix(idx, G, 0.5))[0])(p)["params"]
gn = {k: float(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(v))) ** .5 for k, v in g.items()}
check("gradient reaches torso, aemb, sa1, sa2, du_wa, du_w, du_z1/z2",
      all(gn[k] > 0 for k in ["torso", "aemb", "sa1", "sa2", "du_wa", "du_w", "du_z1", "du_z2"]), {k: f"{v:.2e}" for k, v in gn.items()})
check("logged adv metrics finite", all(np.isfinite(float(info[k])) for k in ["adv_range", "adv_gap", "adv_abs", "W_abs"]))

# bootstrap action a* (target network) = argmax of the target duel_sa advantage
_, _, _, Acn = net.apply(tp, tr.next_obs, method=PairNet.duel_comp_obs)
check("target greedy a* = argmax of target duel_sa A", bool(jnp.all(jnp.argmax(net.apply(tp, tr.next_obs, method=PairNet.scores), -1)
                                                                    == jnp.argmax(Acn, -1))))

# regression: head=duel through the patched code == backup code (same params, same loss and gradient)
bak = "stoix/systems/q_learning/ff_rpair.py.bak_20261104_before_duelsa"
ldr = importlib.machinery.SourceFileLoader("ff_rpair_old", bak)
spec = importlib.util.spec_from_loader("ff_rpair_old", ldr); old = importlib.util.module_from_spec(spec); ldr.exec_module(old)
nd = PairNet(num_actions=A, head="duel", rank=32); no = old.PairNet(num_actions=A, head="duel", rank=32)
pd = nd.init(jax.random.PRNGKey(2), obs[:1]); tpd = nd.init(jax.random.PRNGKey(3), obs[:1])
check("head=duel param tree unchanged (no du_wa)", "du_wa" not in pd["params"]
      and jax.tree_util.tree_structure(pd) == jax.tree_util.tree_structure(no.init(jax.random.PRNGKey(2), obs[:1])))
pmr = rho_weight_matrix(idx, G, 0.5)
_, cn = make_pair_loss(nd, 0.99, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
_, co = old.make_pair_loss(no, 0.99, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
ln, lo = cn(pd, tpd, tr, ar, ar, pmr)[0], co(pd, tpd, tr, ar, ar, pmr)[0]
gnew = jax.grad(lambda q: cn(q, tpd, tr, ar, ar, pmr)[0])(pd); gold = jax.grad(lambda q: co(q, tpd, tr, ar, ar, pmr)[0])(pd)
err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree_util.tree_leaves(gnew), jax.tree_util.tree_leaves(gold)))
check("head=duel loss & gradient identical to pre-patch code", float(ln) == float(lo) and err == 0.0, (float(ln), float(lo), err))
n_sa = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(p)); n_d = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(pd))
print(f"params: duel_sa {n_sa}  duel {n_d}")
print(f"{sum(ok)}/{len(ok)} passed")
