"""check_qcw_cont (2026-11-04): continuous CW-Q (head=qcw) in cw_continuous.py.
JAX_PLATFORMS=cpu .venv/bin/python cwq/checks/check_qcw_cont.py"""
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.ddpg.cw_continuous import QCritic, make_critic_fns, Batch
from stoix.systems.q_learning.same_traj import rho_weight_matrix
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
g = 0.99; rng = np.random.default_rng(1)
m, od, ad = 32, 7, 3
tr = Batch(obs=jnp.array(rng.normal(size=(m, od)), jnp.float32), action=jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32),
           r=jnp.array(rng.normal(size=m), jnp.float32), d=jnp.array(rng.random(m) < 0.3, jnp.float32),
           next_obs=jnp.array(rng.normal(size=(m, od)), jnp.float32))
a_ref = jnp.zeros((m, ad)); a_n = jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32)
net = QCritic(hidden=(32, 32))
x0, a0 = tr.obs[:1], tr.action[:1]
W = rho_weight_matrix(jnp.arange(m), 8, 0.5)
check("rho weight matrix sums to 1, zero diagonal", abs(float(W.sum()) - 1) < 1e-5 and float(jnp.abs(jnp.diag(W)).max()) == 0)

for nc in (1, 2):
    ps = {f"c{k}": net.init(jax.random.PRNGKey(k), x0, a0) for k in range(nc)}
    tps = {f"c{k}": net.init(jax.random.PRNGKey(10 + k), x0, a0) for k in range(nc)}
    closs, aval = make_critic_fns(net, "qcw", g, nc)
    # 1. finite loss, gradients reach every critic parameter (output bias is a pure gauge -> zero grad)
    (L, info), gq = jax.value_and_grad(closs, has_aux=True)(ps, tps, tr, a_ref, a_n, a_ref, W)
    leaves = [(jax.tree_util.keystr(p), x) for p, x in jax.tree_util.tree_flatten_with_path(gq)[0]]
    fin = np.isfinite(float(L)) and all(np.isfinite(np.asarray(x)).all() for _, x in leaves)
    bias_out = [float(jnp.abs(x).max()) for p, x in leaves if "Dense_0" in p and "bias" in p and "MLP" not in p]
    nz = all(float(jnp.abs(x).sum()) > 0 for p, x in leaves if not ("bias" in p and "MLP" not in p))
    check(f"n={nc}: loss/grad finite, non-gauge params get gradient", fin and nz, float(L))
    check(f"n={nc}: output-bias gradient = 0 (level is a gauge, no c)", max(bias_out) < 1e-5 * max(1.0, abs(float(L))) * 1e-3, bias_out)
    # 2. explicit formula with target = online: xi_ij = v_j u_i - v_i u_j, u = r + G min_k Q_k^-(s',a~) - Q
    (L2, _) = closs(ps, ps, tr, a_ref, a_n, a_ref, W)
    qn = jnp.min(jnp.stack([net.apply(ps[c], tr.next_obs, a_n) for c in ps]), 0)
    G = g * (1 - tr.d); v = 1 - G
    Lm = 0.0
    for c in ps:
        u = tr.r + G * qn - net.apply(ps[c], tr.obs, tr.action)
        xi = v[None, :] * u[:, None] - v[:, None] * u[None, :]
        Lm = Lm + 0.5 * jnp.sum(W * jnp.square(xi / (1 - g)))
    check(f"n={nc}: target=online -> loss == 0.5 sum W ((v_j u_i - v_i u_j)/(1-g))^2", abs(float(L2 - Lm)) < 1e-4 * max(1, abs(float(Lm))), (float(L2), float(Lm)))
    # 3. gauge: shift every critic (online and target) by the same constant -> identical loss
    def shift(pp, c):
        return jax.tree_util.tree_map_with_path(
            lambda p, x: x + c if ("bias" in jax.tree_util.keystr(p) and "MLP" not in jax.tree_util.keystr(p)) else x, pp)
    L3, _ = closs(shift(ps, 37.0), shift(tps, 37.0), tr, a_ref, a_n, a_ref, W)
    check(f"n={nc}: invariant to a common shift of Q (online+target, mixed terminations)", abs(float(L3 - L)) < 1e-3 * max(1, abs(float(L))), (float(L), float(L3)))
    # 4. actor read-out = mean Q_1(s, mu(s))
    act = tr.action * 0.5
    check(f"n={nc}: actor value == mean Q_1(s, a)", abs(float(aval(ps["c0"], tr.obs, act, tr, W) - jnp.mean(net.apply(ps["c0"], tr.obs, act)))) < 1e-6)
# 5. no termination + uniform all-pairs: CW-Q gradient == gradient of centred TD loss 0.5 * m/(m-1) * var(u) * 2 ...
#    i.e. sum_ij W (u_i - u_j)^2 / 2 with W uniform = mean over pairs; compare with sample-variance form
tr0 = Batch(obs=tr.obs, action=tr.action, r=tr.r, d=jnp.zeros(m), next_obs=tr.next_obs)
Wu = (1 - jnp.eye(m)) / (m * (m - 1))
ps = {"c0": net.init(jax.random.PRNGKey(0), x0, a0)}
closs, _ = make_critic_fns(net, "qcw", g, 1)
L5, _ = closs(ps, ps, tr0, a_ref, a_n, a_ref, Wu)
u = tr0.r + g * net.apply(ps["c0"], tr0.next_obs, a_n) - net.apply(ps["c0"], tr0.obs, tr0.action)
check("no termination: CW-Q loss == m/(m-1) * var(u) (centred TD / level-free DQN)", abs(float(L5 - m / (m - 1) * jnp.var(u))) < 1e-4, (float(L5), float(m / (m - 1) * jnp.var(u))))
print(f"{sum(ok)}/{len(ok)} passed")
