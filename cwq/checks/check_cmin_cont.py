"""2026-11-08 check for patch_cmin_cont.py (head=qcw, system.min_mode=centered).
JAX_PLATFORMS=cpu .venv/bin/python cwq/checks/check_cmin_cont.py"""
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.ddpg.cw_continuous import QCritic, make_critic_fns, Batch
from stoix.systems.q_learning.same_traj import rho_weight_matrix
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
g = 0.99; rng = np.random.default_rng(1); m, od, ad = 32, 7, 3
tr = Batch(obs=jnp.array(rng.normal(size=(m, od)), jnp.float32), action=jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32),
           r=jnp.array(rng.normal(size=m), jnp.float32), d=jnp.array(rng.random(m) < 0.3, jnp.float32),
           next_obs=jnp.array(rng.normal(size=(m, od)), jnp.float32))
a_ref = jnp.zeros((m, ad)); a_n = jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32)
net = QCritic(hidden=(32, 32)); x0, a0 = tr.obs[:1], tr.action[:1]
W = rho_weight_matrix(jnp.arange(m), 8, 0.5)
ps = {f"c{k}": net.init(jax.random.PRNGKey(k), x0, a0) for k in range(2)}
cl_raw, _ = make_critic_fns(net, "qcw", g, 2)
cl_raw2, _ = make_critic_fns(net, "qcw", g, 2, "raw")
cl_cen, _ = make_critic_fns(net, "qcw", g, 2, "centered")
L_raw, i_raw = cl_raw(ps, ps, tr, a_ref, a_n, a_ref, W)
L_raw2, _ = cl_raw2(ps, ps, tr, a_ref, a_n, a_ref, W)
check("default == raw", float(L_raw) == float(L_raw2))
L_c, i_c = cl_cen(ps, ps, tr, a_ref, a_n, a_ref, W)
qk = jnp.stack([net.apply(ps[c], tr.next_obs, a_n) for c in ps]); cm = qk.mean(1, keepdims=True); mc = jnp.min(qk - cm, 0)
G = g * (1 - tr.d); v = 1 - G; Lm = 0.0
for k, c in enumerate(ps):
    u = tr.r + G * (mc + cm[k, 0]) - net.apply(ps[c], tr.obs, tr.action)
    xi = v[None, :] * u[:, None] - v[:, None] * u[None, :]
    Lm = Lm + 0.5 * jnp.sum(W * jnp.square(xi / (1 - g)))
check("centered: target=online -> explicit formula with qn_k = min_j(Q_j-c_j)+c_k", abs(float(L_c - Lm)) < 1e-4 * max(1, abs(float(Lm))), (float(L_c), float(Lm)))
# shifted twin: critic 1 = critic 0 + 5 -> centered loss == 2 x single-critic loss (each on its own level)
p1 = jax.tree_util.tree_map(lambda x: x, ps["c0"])
p1 = {"params": {**p1["params"], "Dense_0": {**p1["params"]["Dense_0"], "bias": p1["params"]["Dense_0"]["bias"] + 5.0}}}
psh = {"c0": ps["c0"], "c1": p1}
L_sh, i_sh = cl_cen(psh, psh, tr, a_ref, a_n, a_ref, W)
cl1, _ = make_critic_fns(net, "qcw", g, 1)
L1, _ = cl1({"c0": ps["c0"]}, {"c0": ps["c0"]}, tr, a_ref, a_n, a_ref, W)
check("centered: level-shifted twin == 2 x single critic", abs(float(L_sh) - 2 * float(L1)) < 1e-3 * max(1, abs(float(L1))), (float(L_sh), float(L1)))
L_shr, i_shr = cl_raw(psh, psh, tr, a_ref, a_n, a_ref, W)
check("raw: level-shifted twin differs (hijacked), pick0=1", abs(float(L_shr) - 2 * float(L1)) > 1e-3 and float(i_shr["pick0"]) == 1.0, float(L_shr))
check("pick0/pick0c logged", "pick0c" in i_c and "lvl_gap" in i_c)
print(f"{sum(ok)}/{len(ok)} passed")
