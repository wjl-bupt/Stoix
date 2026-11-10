"""2026-11-10 check no_norm in the continuous code (CPU):  JAX_PLATFORMS=cpu .venv/bin/python cwq/checks/check_no_norm_cont.py"""
import sys, importlib.util, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.ddpg.cw_continuous import QCritic, make_critic_fns, Batch
from stoix.systems.q_learning.same_traj import rho_weight_matrix
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
g = 0.99; rng = np.random.default_rng(1); m, od, ad = 32, 7, 3
tr = Batch(obs=jnp.array(rng.normal(size=(m, od)), jnp.float32), action=jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32),
           r=jnp.array(rng.normal(size=m), jnp.float32), d=jnp.array(rng.random(m) < 0.3, jnp.float32),
           next_obs=jnp.array(rng.normal(size=(m, od)), jnp.float32))
a_ref = jnp.zeros((m, ad)); a_n = jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32)
net = QCritic(hidden=(32, 32)); x0, a0 = tr.obs[:1], tr.action[:1]; W = rho_weight_matrix(jnp.arange(m), 8, 0.5)
for nc in (1, 2):
    ps = {f"c{k}": net.init(jax.random.PRNGKey(k), x0, a0) for k in range(nc)}
    tps = {f"c{k}": net.init(jax.random.PRNGKey(10 + k), x0, a0) for k in range(nc)}
    for mm in ("raw", "centered"):
        if mm == "centered" and nc == 1: continue
        gr = {}
        for nn_ in (None, False, True):
            cl, _ = make_critic_fns(net, "qcw", g, nc, mm) if nn_ is None else make_critic_fns(net, "qcw", g, nc, mm, nn_)
            gr[nn_] = flat(jax.grad(lambda p: cl(p, tps, tr, a_ref, a_n, a_ref, W)[0])(ps))
        check(f"cw_continuous n={nc} {mm}: default == no_norm=False", float(jnp.max(jnp.abs(gr[None] - gr[False]))) == 0.0)
        rel = float(jnp.max(jnp.abs(gr[True] - (1 - g) * gr[False])) / jnp.max(jnp.abs((1 - g) * gr[False])))
        check(f"cw_continuous n={nc} {mm}: no_norm grad == (1-g) * grad", rel < 1e-4, f"{rel:.1e}")
spec = importlib.util.spec_from_file_location("mo", "/home/junweiluo/Stoix/cwq/mujoco_gym/mujoco_offpolicy.py")
mo = importlib.util.module_from_spec(spec); spec.loader.exec_module(mo)
q = mo.QNet(); b = {"obs": tr.obs, "action": tr.action, "r": tr.r, "d": tr.d, "next_obs": tr.next_obs}
qp = tuple(q.init(jax.random.PRNGKey(k), x0, a0) for k in range(2)); tqp = tuple(q.init(jax.random.PRNGKey(20 + k), x0, a0) for k in range(2))
for mm in ("raw", "centered"):
    g0 = flat(jax.grad(lambda p: mo.critic_loss_fn(q, p, tqp, b, a_n, g, "qcw", W, mm)[0])(qp))
    g1 = flat(jax.grad(lambda p: mo.critic_loss_fn(q, p, tqp, b, a_n, g, "qcw", W, mm, 1)[0])(qp))
    rel = float(jnp.max(jnp.abs(g1 - (1 - g) * g0)) / jnp.max(jnp.abs((1 - g) * g0)))
    check(f"mujoco_offpolicy {mm}: no_norm grad == (1-g) * grad", rel < 1e-4, f"{rel:.1e}")
print(f"{sum(ok)}/{len(ok)} passed")
