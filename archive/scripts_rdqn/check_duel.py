"""Checks for head=duel (CW-Duel) in ff_rpair.py, 2026-11-02.  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_duel.py
D1  Borda read-out: scores = A/(A-1) Ac  equals  mean_{b!=a} D((s,a),(s,b)) computed by duel_D; argmax = argmax Ac.
D2  antisymmetry D(x,y) = -D(y,x), D(x,x) = 0; same-state difference D((s,a),(s,b)) = Ac(s,a) - Ac(s,b).
D3  all-pairs loss / delta equal a brute-force per-pair computation with the generic cross formula (rank 32 and 0).
D4  gradient of the matrix loss equals the gradient of the brute-force per-pair loss.
D5  Prop. 3 (rank 0, consistent class Q = w + Ac): (1-g) delta_ij = v_j u_i - v_i u_j with the DQN TD error
    u_i = r_i + g(1-d_i) Q^-(s'_i, a*_i) - Q(s_i, a_i)  (current state online, next state target).
D6  identifiability (Thm. 2c) on the network class: two parameter sets with equal D on all CROSS-state pairs
    of a batch have equal same-state differences -- checked as: same-state differences are recoverable
    from cross-state D values by D((s,a),(t,b)) - D((s,a'),(t,b)).
D7  pair-type formulas: both continue / x ends / both end.
"""
import sys
sys.path.insert(0, "/home/junweiluo/Stoix")
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.dqn_types import Transition
jax.config.update("jax_enable_x64", True)

ok = tot = 0
def chk(name, cond, extra=""):
    global ok, tot
    tot += 1; ok += bool(cond); print(("PASS " if cond else "FAIL ") + name + " " + extra)

A, g, m = 6, 0.99, 8
key = jax.random.PRNGKey(0)
obs_shape = (10, 10, 4)

def rand_batch(key, m, pterm=0.3):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    obs = jax.random.bernoulli(k1, 0.3, (m,) + obs_shape).astype(jnp.float64)
    nxt = jax.random.bernoulli(k2, 0.3, (m,) + obs_shape).astype(jnp.float64)
    act = jax.random.randint(k3, (m,), 0, A)
    rew = jax.random.normal(k4, (m,)).astype(jnp.float64)
    done = jax.random.bernoulli(k5, pterm, (m,))
    return Transition(obs=obs, action=act, reward=rew, done=done, next_obs=nxt, info={})

def perturb(params, key, scale=0.02):
    leaves, tdef = jax.tree_util.tree_flatten(params)
    ks = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(tdef, [l + scale * jax.random.normal(k, l.shape, l.dtype) for l, k in zip(leaves, ks)])

for rank in (32, 0):
    net = PairNet(num_actions=A, head="duel", rank=rank)
    tr = rand_batch(jax.random.PRNGKey(1 + rank), m)
    p = net.init(jax.random.PRNGKey(2), tr.obs[:1])
    p = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), p)
    p = perturb(p, jax.random.PRNGKey(3)); tp = perturb(p, jax.random.PRNGKey(4))     # target != online
    ap = lambda pp, meth, *a: net.apply(pp, *a, method=meth)                          # noqa: E731

    # D1 / D2
    S = tr.obs
    sc = ap(p, PairNet.scores, S)
    Ac = ap(p, PairNet.duel_comp_obs, S)[3]
    borda = np.zeros((m, A))
    for a in range(A):
        tot_ = 0.0
        for b in range(A):
            if b == a: continue
            tot_ = tot_ + ap(p, PairNet.duel_D, S, jnp.full((m,), a), S, jnp.full((m,), b))
        borda[:, a] = np.asarray(tot_) / (A - 1)
    chk(f"D1 r={rank} Borda closed form", np.abs(borda - np.asarray(sc)).max() < 1e-10, f"{np.abs(borda - np.asarray(sc)).max():.1e}")
    chk(f"D1 r={rank} argmax = argmax Ac", (np.argmax(np.asarray(sc), -1) == np.argmax(np.asarray(Ac), -1)).all())
    a1 = jax.random.randint(jax.random.PRNGKey(7), (m,), 0, A); a2 = jax.random.randint(jax.random.PRNGKey(8), (m,), 0, A)
    perm = jnp.roll(jnp.arange(m), 3)
    dxy = ap(p, PairNet.duel_D, S, a1, S[perm], a2); dyx = ap(p, PairNet.duel_D, S[perm], a2, S, a1)
    dxx = ap(p, PairNet.duel_D, S, a1, S, a1)
    same = ap(p, PairNet.duel_D, S, a1, S, a2)
    chk(f"D2 r={rank} antisym / D(x,x)=0 / same-state = A diff",
        max(np.abs(dxy + dyx).max(), np.abs(dxx).max(),
            np.abs(same - (Ac[jnp.arange(m), a1] - Ac[jnp.arange(m), a2])).max()) < 1e-10)

    # D3 / D4: brute force
    loss_fn, compute = make_pair_loss(net, g, "cross", 0.33, 1e9, return_delta=True, all_pairs=True)
    (L, info) = compute(p, tp, tr, jnp.arange(m))
    k_ = (1 - g) / g
    def brute(pp):
        sg = jax.lax.stop_gradient
        Acn = ap(tp, PairNet.duel_comp_obs, tr.next_obs)[3]; astar = jnp.argmax(Acn, -1)
        Dsg = lambda ox, axx, oy, ayy, px, py: None  # placeholder (not used)
        tot_l = 0.0; dl = np.zeros((m, m))
        rows = []
        for i in range(m):
            for j in range(m):
                if i == j: continue
                si, sj, ni, nj = tr.obs[i:i+1], tr.obs[j:j+1], tr.next_obs[i:i+1], tr.next_obs[j:j+1]
                ai, aj, si_, sj_ = tr.action[i:i+1], tr.action[j:j+1], astar[i:i+1], astar[j:j+1]
                D_on = ap(pp, PairNet.duel_D, si, ai, sj, aj)[0]
                # mixed-network terms: components from the indicated network
                def comp(pq, o): return ap(pq, PairNet.duel_comp_obs, o)
                def Dmix(cx, ax_, cy, ay_):
                    wx, z1x, z2x, Ax_ = cx; wy, z1y, z2y, Ay_ = cy
                    return (wx - wy + jnp.sum(z1x * z2y, -1) - jnp.sum(z2x * z1y, -1)
                            + Ax_[0, ax_[0]] - Ay_[0, ay_[0]])[0]
                cs_i = jax.tree_util.tree_map(sg, comp(pp, si)); cs_j = jax.tree_util.tree_map(sg, comp(pp, sj))
                cn_i, cn_j = comp(tp, ni), comp(tp, nj)
                M = Dmix(cn_i, si_, cn_j, sj_); Amx = Dmix(cn_i, si_, cs_j, aj); Bmn = Dmix(cs_i, ai, cn_j, sj_)
                rx, ry = tr.reward[i], tr.reward[j]
                Gx, Gy = g * (1 - tr.done[i]), g * (1 - tr.done[j]); vx, vy = 1 - Gx, 1 - Gy
                W1 = k_ * Gx * Gy; W2 = Gx * (vy - k_ * Gy); W3 = Gy * (vx - k_ * Gx); W4 = -vy - Gy * (vx - k_ * Gx)
                xi = vy * rx - vx * ry + W1 * M + W2 * Amx + W3 * Bmn + W4 * sg(D_on)
                dlt = sg(xi) / (1 - g)
                rows.append(dlt)
                tot_l = tot_l + 0.5 * jnp.square(sg(D_on + dlt) - D_on)
        return tot_l / (m * (m - 1)), jnp.stack(rows)
    Lb, rows = brute(p)
    dmat = np.asarray(info["_delta"]); off = ~np.eye(m, dtype=bool)
    chk(f"D3 r={rank} loss = brute", abs(float(L) - float(Lb)) < 1e-5 * max(1.0, abs(float(Lb))), f"{float(L):.6f} vs {float(Lb):.6f}")
    chk(f"D3 r={rank} delta = brute", np.abs(dmat[off] - np.asarray(rows)).max() < 1e-5 * max(1.0, np.abs(dmat).max()), f"{np.abs(dmat[off] - np.asarray(rows)).max():.1e}")
    gm = jax.grad(lambda pp: compute(pp, tp, tr, jnp.arange(m))[0])(p)
    gb = jax.grad(lambda pp: brute(pp)[0])(p)
    err = max(float(jnp.abs(a_ - b_).max()) for a_, b_ in zip(jax.tree_util.tree_leaves(gm), jax.tree_util.tree_leaves(gb)))
    gsc = max(float(jnp.abs(b_).max()) for b_ in jax.tree_util.tree_leaves(gb))
    chk(f"D4 r={rank} grad = brute", err < 1e-5 * max(1.0, gsc), f"{err:.1e} (scale {gsc:.1e})")

    if rank == 0:   # D5 Prop. 3
        w_on, _, _, Ac_on = ap(p, PairNet.duel_comp_obs, tr.obs)
        w_n, _, _, Ac_n = ap(tp, PairNet.duel_comp_obs, tr.next_obs)
        astar = jnp.argmax(Ac_n, -1)
        Q = w_on + Ac_on[jnp.arange(m), tr.action]; Qn = w_n + Ac_n[jnp.arange(m), astar]
        u = tr.reward + g * (1 - tr.done) * Qn - Q; v = 1 - g * (1 - tr.done)
        pred = (v[None, :] * u[:, None] - v[:, None] * u[None, :]) / (1 - g)
        e5 = np.abs(np.asarray(pred)[off] - dmat[off]).max()
        chk("D5 r=0 Prop.3 xi_ij = v_j u_i - v_i u_j", e5 < 1e-5 * max(1.0, np.abs(dmat).max()), f"{e5:.1e} (scale {np.abs(dmat).max():.1e})")
        rowsum = (dmat * off).sum(1); cb = float(u.sum() / v.sum())
        clos = (np.asarray(u) * float(v.sum()) - np.asarray(v) * float(u.sum())) / (1 - g)
        # full-pair row sum excludes j=i: sum_{j!=i} (v_j u_i - v_i u_j) = u_i Sv - v_i Su (the i=j term is 0)
        chk("D5 r=0 row-sum closed form", np.abs(rowsum - clos).max() < 1e-5 * max(1.0, np.abs(clos).max()), f"{np.abs(rowsum - clos).max():.1e} c_batch={cb:.3f}")

    # D6: same-state differences recovered from cross-state D values
    t_obs = tr.obs[perm]; b = jnp.zeros((m,), int)
    rec = ap(p, PairNet.duel_D, S, a1, t_obs, b) - ap(p, PairNet.duel_D, S, a2, t_obs, b)
    chk(f"D6 r={rank} cross-state D determines A diff", np.abs(np.asarray(rec - same)).max() < 1e-10)

# D7 pair types (rank 32): compare with closed pair-type formulas
net = PairNet(num_actions=A, head="duel", rank=32)
tr = rand_batch(jax.random.PRNGKey(11), m, pterm=0.5)
p = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), net.init(jax.random.PRNGKey(2), tr.obs[:1]))
p = perturb(p, jax.random.PRNGKey(5)); tp = perturb(p, jax.random.PRNGKey(6))
_, compute = make_pair_loss(net, g, "cross", 0.33, 1e9, return_delta=True, all_pairs=True)
_, info = compute(p, tp, tr, jnp.arange(m)); dmat = np.asarray(info["_delta"]) * (1 - g)
c_on = net.apply(p, tr.obs, method=PairNet.duel_comp_obs); c_n = net.apply(tp, tr.next_obs, method=PairNet.duel_comp_obs)
ast = np.asarray(jnp.argmax(c_n[3], -1)); act = np.asarray(tr.action)
def Dc(c1, i, a1_, c2, j, a2_):
    return float(c1[0][i] - c2[0][j] + c1[1][i] @ c2[2][j] - c1[2][i] @ c2[1][j] + c1[3][i, a1_] - c2[3][j, a2_])
d = np.asarray(tr.done); r = np.asarray(tr.reward); errs = []
for i in range(m):
    for j in range(m):
        if i == j: continue
        D = Dc(c_on, i, act[i], c_on, j, act[j])
        if not d[i] and not d[j]:
            ref = (1 - g) * (r[i] - r[j] + g * Dc(c_n, i, ast[i], c_n, j, ast[j]) - D)
        elif d[i] and not d[j]:
            ref = (1 - g) * r[i] - r[j] + g * Dc(c_on, i, act[i], c_n, j, ast[j]) - D
        elif d[j] and not d[i]:
            ref = r[i] - (1 - g) * r[j] - g * Dc(c_on, j, act[j], c_n, i, ast[i]) - D
        else:
            ref = r[i] - r[j] - D
        errs.append(abs(ref - dmat[i, j]))
chk("D7 pair-type formulas", max(errs) < 1e-5 * max(1.0, np.abs(dmat).max()), f"{max(errs):.1e} (scale {np.abs(dmat).max():.1e})")
print(f"{ok}/{tot} passed")
