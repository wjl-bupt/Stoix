"""Checks for head=duel_mlp (CW-Duel, MLP state-pair head, K partners), 2026-11-02.
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_duel_mlp.py
E1 Borda read-out = mean_{b!=a} dm_D((s,a),(s,b)); argmax = argmax Ac.    E2 antisymmetry, D(x,x)=0, same-state = A diff.
E3 loss / delta of duel_mlp_pairs equal a brute-force per-pair computation with the generic cross formula,
   current-state parts of Amax/Bmin from the TARGET network.           E4 gradient equals brute-force gradient.
E5 pair-type closed forms.   E6 n_partners=64 shifts are 64 distinct non-zero offsets (m=512), each i appears 64x as x and 64x as y.
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
A, g, m = 6, 0.99, 10
obs_shape = (10, 10, 4)
def rand_batch(key, m, pterm=0.3):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    return Transition(obs=jax.random.bernoulli(k1, 0.3, (m,) + obs_shape).astype(jnp.float64), action=jax.random.randint(k3, (m,), 0, A),
                      reward=jax.random.normal(k4, (m,)).astype(jnp.float64), done=jax.random.bernoulli(k5, pterm, (m,)),
                      next_obs=jax.random.bernoulli(k2, 0.3, (m,) + obs_shape).astype(jnp.float64), info={})
def perturb(params, key, scale=0.05):
    leaves, tdef = jax.tree_util.tree_flatten(params); ks = jax.random.split(key, len(leaves))
    return jax.tree_util.tree_unflatten(tdef, [l + scale * jax.random.normal(k, l.shape, l.dtype) for l, k in zip(leaves, ks)])
net = PairNet(num_actions=A, head="duel_mlp")
tr = rand_batch(jax.random.PRNGKey(1), m)
p = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), net.init(jax.random.PRNGKey(2), tr.obs[:1]))
p = perturb(p, jax.random.PRNGKey(3)); tp = perturb(p, jax.random.PRNGKey(4))
ap = lambda pp, meth, *a: net.apply(pp, *a, method=meth)  # noqa
S = tr.obs
sc = ap(p, PairNet.scores, S); Ac = ap(p, PairNet.dm_comp_obs, S)[1]
borda = np.zeros((m, A))
for a in range(A):
    t_ = 0.0
    for b in range(A):
        if b != a: t_ = t_ + ap(p, PairNet.dm_D, S, jnp.full((m,), a), S, jnp.full((m,), b))
    borda[:, a] = np.asarray(t_) / (A - 1)
chk("E1 Borda closed form", np.abs(borda - np.asarray(sc)).max() < 1e-10, f"{np.abs(borda - np.asarray(sc)).max():.1e}")
chk("E1 argmax = argmax Ac", (np.argmax(np.asarray(sc), -1) == np.argmax(np.asarray(Ac), -1)).all())
a1 = jax.random.randint(jax.random.PRNGKey(7), (m,), 0, A); a2 = jax.random.randint(jax.random.PRNGKey(8), (m,), 0, A)
perm = jnp.roll(jnp.arange(m), 3)
dxy = ap(p, PairNet.dm_D, S, a1, S[perm], a2); dyx = ap(p, PairNet.dm_D, S[perm], a2, S, a1)
dxx = ap(p, PairNet.dm_D, S, a1, S, a1); same = ap(p, PairNet.dm_D, S, a1, S, a2)
chk("E2 antisym / D(x,x)=0 / same-state = A diff", max(np.abs(dxy + dyx).max(), np.abs(dxx).max(), np.abs(same - (Ac[jnp.arange(m), a1] - Ac[jnp.arange(m), a2])).max()) < 1e-10)
K = 3
loss_fn, compute = make_pair_loss(net, g, "cross", 0.33, 1e9, return_delta=True, n_partners=K, all_pairs=False)
shift = jnp.int32(4)
L, info = loss_fn(p, tp, tr, shift)
shifts = 1 + (shift - 1 + jnp.arange(K) * ((m - 1) // K)) % (m - 1)
ix = np.asarray(jnp.tile(jnp.arange(m), K)); iy = np.asarray(((jnp.arange(m)[None, :] + shifts[:, None]) % m).reshape(-1))
k_ = (1 - g) / g
def brute(pp):
    sg = jax.lax.stop_gradient
    Acn = ap(tp, PairNet.dm_comp_obs, tr.next_obs)[1]; astar = jnp.argmax(Acn, -1)
    tot_l = 0.0; rows = []
    for i, j in zip(ix, iy):
        si, sj, ni, nj = tr.obs[i:i+1], tr.obs[j:j+1], tr.next_obs[i:i+1], tr.next_obs[j:j+1]
        ai, aj, si_, sj_ = tr.action[i:i+1], tr.action[j:j+1], astar[i:i+1], astar[j:j+1]
        D_on = ap(pp, PairNet.dm_D, si, ai, sj, aj)[0]
        M = ap(tp, PairNet.dm_D, ni, si_, nj, sj_)[0]
        Amx = ap(tp, PairNet.dm_D, ni, si_, sj, aj)[0]       # current-state side from the target network
        Bmn = ap(tp, PairNet.dm_D, si, ai, nj, sj_)[0]
        rx, ry = tr.reward[i], tr.reward[j]
        Gx, Gy = g * (1 - tr.done[i]), g * (1 - tr.done[j]); vx, vy = 1 - Gx, 1 - Gy
        W1 = k_ * Gx * Gy; W2 = Gx * (vy - k_ * Gy); W3 = Gy * (vx - k_ * Gx); W4 = -vy - Gy * (vx - k_ * Gx)
        xi = vy * rx - vx * ry + W1 * M + W2 * Amx + W3 * Bmn + W4 * sg(D_on)
        dlt = sg(xi) / (1 - g); rows.append(dlt)
        tot_l = tot_l + 0.5 * jnp.square(sg(D_on + dlt) - D_on)
    return tot_l / len(ix), jnp.stack(rows)
Lb, rows = brute(p)
chk("E3 loss = brute", abs(float(L) - float(Lb)) < 1e-5 * max(1.0, abs(float(Lb))), f"{float(L):.6f} vs {float(Lb):.6f}")
dd = np.abs(np.asarray(info["_delta"]) - np.asarray(rows)).max(); ds = np.abs(np.asarray(rows)).max()
chk("E3 delta = brute", dd < 1e-5 * max(1.0, ds), f"{dd:.1e} (scale {ds:.1e})")
gm = jax.grad(lambda pp: loss_fn(pp, tp, tr, shift)[0])(p); gb = jax.grad(lambda pp: brute(pp)[0])(p)
err = max(float(jnp.abs(a_ - b_).max()) for a_, b_ in zip(jax.tree_util.tree_leaves(gm), jax.tree_util.tree_leaves(gb)))
gsc = max(float(jnp.abs(b_).max()) for b_ in jax.tree_util.tree_leaves(gb))
chk("E4 grad = brute", err < 1e-5 * max(1.0, gsc), f"{err:.1e} (scale {gsc:.1e})")
# E5 pair types
dm = np.asarray(info["_delta"]) * (1 - g); d = np.asarray(tr.done); r = np.asarray(tr.reward)
h_on, Ac_on = ap(p, PairNet.dm_comp_obs, tr.obs); h_t, Ac_t = ap(tp, PairNet.dm_comp_obs, tr.obs); h_n, Ac_n = ap(tp, PairNet.dm_comp_obs, tr.next_obs)
ast = np.asarray(jnp.argmax(Ac_n, -1)); act = np.asarray(tr.action)
def W(pp, hx, hy): return float(ap(pp, PairNet.dm_W, hx[None], hy[None])[0])
errs = []
for q, (i, j) in enumerate(zip(ix, iy)):
    D = W(p, h_on[i], h_on[j]) + float(Ac_on[i, act[i]] - Ac_on[j, act[j]])
    if not d[i] and not d[j]:
        ref = (1 - g) * (r[i] - r[j] + g * (W(tp, h_n[i], h_n[j]) + float(Ac_n[i, ast[i]] - Ac_n[j, ast[j]])) - D)
    elif d[i] and not d[j]:
        ref = (1 - g) * r[i] - r[j] + g * (W(tp, h_t[i], h_n[j]) + float(Ac_t[i, act[i]] - Ac_n[j, ast[j]])) - D
    elif d[j] and not d[i]:
        ref = r[i] - (1 - g) * r[j] + g * (W(tp, h_n[i], h_t[j]) + float(Ac_n[i, ast[i]] - Ac_t[j, act[j]])) - D
    else:
        ref = r[i] - r[j] - D
    errs.append(abs(ref - dm[q]))
chk("E5 pair-type formulas", max(errs) < 1e-5 * max(1.0, np.abs(dm).max()), f"{max(errs):.1e}")
# E6 shifts for m=512, K=64
m2, K2 = 512, 64
for sh in (1, 100, 511):
    shifts2 = 1 + (sh - 1 + np.arange(K2) * ((m2 - 1) // K2)) % (m2 - 1)
    ix2 = np.tile(np.arange(m2), K2); iy2 = ((np.arange(m2)[None, :] + shifts2[:, None]) % m2).reshape(-1)
    chk(f"E6 shift={sh}: 64 distinct non-zero offsets, no self-pairs, each i 64x as x / 64x as y",
        len(set(shifts2.tolist())) == K2 and (shifts2 > 0).all() and (shifts2 < m2).all() and (ix2 != iy2).all()
        and (np.bincount(ix2, minlength=m2) == K2).all() and (np.bincount(iy2, minlength=m2) == K2).all())
print(f"{ok}/{tot} passed")
