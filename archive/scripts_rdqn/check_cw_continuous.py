"""check_cw_continuous (2026-11-04).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_cw_continuous.py"""
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.ddpg.cw_continuous import (QCritic, PairCritic, DuelCritic, make_critic_fns, cross_xi, Batch, wmat)
from stoix.systems.q_learning.same_traj import rho_weight_matrix
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
g = 0.99; rng = np.random.default_rng(0)
# 1. cross-weighted residual special cases (ff_rpair docstring)
rx, ry, M, Am, Bm, D = rng.normal(size=6)
c1 = cross_xi(D, M, Am, Bm, rx, ry, 0.0, 0.0, g); c2 = cross_xi(D, M, Am, Bm, rx, ry, 1.0, 0.0, g); c3 = cross_xi(D, M, Am, Bm, rx, ry, 1.0, 1.0, g)
check("xi both continue = (1-g)[r_x - r_y + g M - D]", np.isclose(c1, (1 - g) * (rx - ry + g * M - D)))
check("xi x ends = (1-g) r_x - r_y + g Bmin - D", np.isclose(c2, (1 - g) * rx - ry + g * Bm - D))
check("xi both end = r_x - r_y - D", np.isclose(c3, rx - ry - D))
# random batch
m, od, ad = 32, 7, 3
tr = Batch(obs=jnp.array(rng.normal(size=(m, od)), jnp.float32), action=jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32),
           r=jnp.array(rng.normal(size=m), jnp.float32), d=jnp.array(rng.random(m) < 0.2, jnp.float32),
           next_obs=jnp.array(rng.normal(size=(m, od)), jnp.float32))
a_ref = jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32); a_n_ref = jnp.array(rng.uniform(-1, 1, (m, ad)), jnp.float32)
a_n = jnp.clip(a_n_ref + 0.1, -1, 1)
x0, a0 = tr.obs[:1], tr.action[:1]
for head, net in (("q", QCritic(hidden=(32, 32))), ("pair", PairCritic(torso=(32,), enc=(32,), pair_hidden=(32,))),
                  ("duel", DuelCritic(torso=(32,), f_hidden=(32,), rank=4))):
    for nc in (1, 2):
        ps = {f"c{k}": net.init(jax.random.PRNGKey(k), x0, a0) for k in range(nc)}
        tps = {f"c{k}": net.init(jax.random.PRNGKey(10 + k), x0, a0) for k in range(nc)}
        closs, aval = make_critic_fns(net, head, g, nc)
        if head == "pair":
            ix = jnp.tile(jnp.arange(m), 3); iy = (ix + jnp.repeat(jnp.arange(1, 4), m)) % m
            pairs = (ix, iy, jnp.full(ix.shape, 1.0 / ix.shape[0]))
        elif head == "duel":
            pairs = rho_weight_matrix(jnp.arange(m), 8, 0.5)
        else:
            pairs = None
        (L, info), gq = jax.value_and_grad(closs, has_aux=True)(ps, tps, tr, a_ref, a_n, a_n_ref, pairs)
        fin = np.isfinite(float(L)) and all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(gq))
        # gauge parameters that cancel by construction: duel f_out bias (A = f(s,a) - f(s,a_ref)), w_out bias (w(s)-w(t))
        leaves = [(jax.tree_util.keystr(pth), x) for pth, x in jax.tree_util.tree_flatten_with_path(gq["c0"])[0]]
        nz = all(float(jnp.abs(x).sum()) > 0 for pth, x in leaves
                 if not (head == "duel" and ("f_out" in pth or "w_out" in pth) and "bias" in pth))
        check(f"{head} n_critics={nc}: critic loss/grad finite, grads reach all critic-1 params", fin and nz, float(L))
        ga = jax.grad(lambda act: aval(ps["c0"], tr.obs, act, tr, pairs))(tr.action * 0.3)
        check(f"{head}: actor read-out has nonzero action gradient", float(jnp.abs(ga).sum()) > 0)
        if head == "q":
            qn = jnp.stack([net.apply(tps[c], tr.next_obs, a_n) for c in ps])
            y = tr.r + g * (1 - tr.d) * qn.min(0)
            Lm = sum(0.5 * jnp.mean(jnp.square(y - net.apply(ps[c], tr.obs, tr.action))) for c in ps)
            check(f"q n={nc}: loss == 0.5 mean (r + g(1-d) min_k Q_k^- - Q)^2 summed over critics", abs(float(L - Lm)) < 1e-5)
        if head == "pair" and nc == 2:
            from stoix.systems.ddpg.cw_continuous import cross_xi as cx
            Ts = []
            Ds = []
            for c in ("c0", "c1"):
                eo = net.apply(ps[c], tr.obs, tr.action, method=PairCritic.encode); et = net.apply(tps[c], tr.obs, tr.action, method=PairCritic.encode)
                en = net.apply(tps[c], tr.next_obs, a_n, method=PairCritic.encode)
                Dc = net.apply(ps[c], eo[ix], eo[iy], method=PairCritic.D)
                xi = cx(Dc, net.apply(tps[c], en[ix], en[iy], method=PairCritic.D), net.apply(tps[c], en[ix], et[iy], method=PairCritic.D),
                        net.apply(tps[c], et[ix], en[iy], method=PairCritic.D), tr.r[ix], tr.r[iy], tr.d[ix], tr.d[iy], g)
                Ts.append(Dc + xi / (1 - g)); Ds.append(Dc)
            T = jnp.minimum(Ts[0], Ts[1]); w = pairs[2]
            Lm = sum(0.5 * jnp.sum(w * jnp.square(T - Dc)) for Dc in Ds)
            check("pair TD3: both critics regress to min_k T_k", abs(float(L - Lm)) < 1e-4 * max(1, abs(float(Lm))), (float(L), float(Lm)))
        if head == "duel":
            p0 = ps["c0"]
            c = net.apply(p0, tr.obs, method=DuelCritic.state)
            A = net.apply(p0, c[0], tr.action, method=DuelCritic.f) - net.apply(p0, c[0], a_ref, method=DuelCritic.f)
            Dm = wmat(c, c) + A[:, None] - A[None, :]
            check("duel D antisymmetric (D_ij = -D_ji, D_ii = 0)", float(jnp.abs(Dm + Dm.T).max()) < 1e-5)
            An = net.apply(p0, c[0], a_ref, method=DuelCritic.f) - net.apply(p0, c[0], a_ref, method=DuelCritic.f)
            check("duel A(s, a_ref(s)) = 0 (DDPG: next-state advantage term vanishes)", float(jnp.abs(An).max()) == 0.0)
print(f"{sum(ok)}/{len(ok)} passed")
