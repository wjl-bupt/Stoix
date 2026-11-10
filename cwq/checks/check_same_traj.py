"""Checks for same_traj.same_traj_batch (2026-11-03).  JAX_PLATFORMS=cpu .venv/bin/python cwq/checks/check_same_traj.py"""
import numpy as np, jax, jax.numpy as jnp, flashbax as fbx
from stoix.systems.q_learning.dqn_types import Transition
from stoix.systems.q_learning.same_traj import same_traj_batch

ok = []
def check(name, cond, extra=""):
    ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, extra)

def build(E, L, steps, warm, p_done, seed):
    rng = np.random.default_rng(seed)
    dummy = Transition(obs=jnp.zeros(3), action=jnp.zeros((), int), reward=jnp.zeros(()), done=jnp.zeros((), bool),
                       next_obs=jnp.zeros(3), info={"x": 0.0})
    buf = fbx.make_item_buffer(max_length=L, min_length=8, sample_batch_size=8, add_batches=True, add_sequences=True)
    st = buf.init(dummy)
    ep = np.arange(E) * 100000; t = np.zeros(E); nxt_ep = E * 100000
    rows = []
    def one_step():
        nonlocal nxt_ep
        d = rng.random(E) < p_done
        obs = np.stack([np.arange(E), t.copy(), ep.copy()], 1)
        nobs = np.stack([np.arange(E), t + 1, ep.copy()], 1)
        rows.append((obs, nobs, d))
        for e in range(E):
            if d[e]:
                ep[e] = nxt_ep; nxt_ep += 1; t[e] = 0
            else:
                t[e] += 1
        return obs, nobs, d
    def tr(lst):
        o = np.stack([x[0] for x in lst]); n = np.stack([x[1] for x in lst]); d = np.stack([x[2] for x in lst])
        T = o.shape[0]
        return Transition(obs=jnp.array(o, jnp.float32), action=jnp.zeros((T, E), int), reward=jnp.zeros((T, E)),
                          done=jnp.array(d), next_obs=jnp.array(n, jnp.float32), info={"x": jnp.zeros((T, E))})
    st = buf.add(st, tr([one_step() for _ in range(warm)]))
    for _ in range(steps):
        st = buf.add(st, tr([one_step()]))
    return st, rows

def run(E, L, steps, warm, m, G, p_done, n_keys=400, seed=0, tag=""):
    st, rows = build(E, L, steps, warm, p_done, seed)
    total = (warm + steps) * E
    n_valid = min(total, L)
    # ground truth list of valid items (newest n_valid writes)
    flat = []
    for k, (o, n, d) in enumerate(rows):
        for e in range(E):
            flat.append((tuple(o[e]), bool(d[e])))
    valid_items = flat[-n_valid:]
    vset = {it[0] for it in valid_items}
    ep_items = {}
    for it in valid_items:
        ep_items.setdefault(it[0][2], set()).add(it[0])
    f = jax.jit(lambda s, k: same_traj_batch(s, k, m, G, E))
    bad_ep = bad_self = bad_valid = 0; anchors = []; chain = []; partner_cnt = {}; n_pairs = 0; masked = 0
    for i in range(n_keys):
        b, ix, iy, w, stt = f(st, jax.random.PRNGKey(seed * 1000 + i))
        o = np.asarray(b.obs); ix = np.asarray(ix); iy = np.asarray(iy); w = np.asarray(w)
        chain.append(float(stt["st_chain_ok"]))
        for row in o:
            if tuple(row) not in vset: bad_valid += 1
        ng = m // G
        for g in range(ng):
            a = tuple(o[g * G]); anchors.append(a)
            for s in range(1, G):
                pt = tuple(o[g * G + s])
                if pt[2] != a[2] or pt[0] != a[0]: bad_ep += 1
                if pt == a and len(ep_items[a[2]]) > 1: bad_self += 1
                if len(ep_items[a[2]]) > 1:
                    partner_cnt.setdefault(a, []).append(pt)
        n_pairs += len(ix); masked += int((w == 0).sum())
        assert abs(w.sum() - 1) < 1e-5 or w.sum() == 0
    check(f"{tag} all items valid", bad_valid == 0, bad_valid)
    check(f"{tag} partners in same env+episode", bad_ep == 0, bad_ep)
    check(f"{tag} partner != anchor (non-singleton)", bad_self == 0, bad_self)
    check(f"{tag} chain_ok == 1", min(chain) == 1.0, (min(chain), np.mean(chain)))
    # anchor uniformity over valid items: chi-square on episode-position buckets via item index
    idx = {it[0]: j for j, it in enumerate(valid_items)}
    a_idx = np.array([idx[a] for a in anchors]); nb = 20
    h = np.histogram(a_idx, bins=nb, range=(0, n_valid))[0]; exp_ = len(a_idx) / nb
    chi = ((h - exp_) ** 2 / exp_).sum()
    check(f"{tag} anchor uniform (chi2 df19 < 43.8)", chi < 43.8, round(chi, 1))
    # partner uniform within episode: relative position histogram pooled
    rel = []; prng = np.random.default_rng(7)
    for a, pts in partner_cnt.items():
        items = sorted(ep_items[a[2]], key=lambda x: x[1]); others = [x for x in items if x != a]
        pos = {x: j for j, x in enumerate(others)}
        for pt in pts: rel.append((pos[pt] + prng.random()) / len(others))   # randomized PIT: exactly U(0,1) iff uniform
    h = np.histogram(rel, bins=10, range=(0, 1))[0]; exp_ = len(rel) / 10
    chi = ((h - exp_) ** 2 / exp_).sum()
    check(f"{tag} partner ~uniform within episode (pooled, chi2 df9 < 27.9)", chi < 27.9, round(chi, 1))
    print(f"   {tag}: masked pair frac {masked / n_pairs:.3f}")

run(E=8, L=1003, steps=300, warm=40, m=64, G=8, p_done=0.05, tag="wrapped")
run(E=8, L=5000, steps=100, warm=40, m=64, G=8, p_done=0.05, tag="not-full")
run(E=16, L=777, steps=200, warm=5, m=32, G=4, p_done=0.2, tag="short-eps")

# random-groups control: partners uniform over valid items, not tied to the anchor's episode
st, rows = build(8, 1003, 300, 40, 0.05, 0)
f = jax.jit(lambda s_, k: same_traj_batch(s_, k, 64, 8, 8, random_groups=True))
vals = set(); same_ep = tot = 0; pos_list = []
flat = [tuple(o[e]) for (o, n, d) in rows for e in range(8)][-1003:]
idxmap = {x: j for j, x in enumerate(flat)}
for i in range(300):
    b, ix_, iy_, w_, _ = f(st, jax.random.PRNGKey(i)); o = np.asarray(b.obs)
    for g in range(8):
        a = tuple(o[g * 8])
        for s_ in range(1, 8):
            pt = tuple(o[g * 8 + s_]); tot += 1; same_ep += pt[2] == a[2]; pos_list.append(idxmap[pt])
h = np.histogram(pos_list, bins=20, range=(0, 1003))[0]; e_ = len(pos_list) / 20
chi = ((h - e_) ** 2 / e_).sum()
check("random groups: partners uniform over buffer (chi2 df19 < 43.8)", chi < 43.8, round(chi, 1))
check("random groups: same-episode partner rate small (< 0.05)", same_ep / tot < 0.05, round(same_ep / tot, 4))

# mix: same-trajectory groups unchanged + n_cross partners from other trajectories (anchor <-> cross, both orders)
f = jax.jit(lambda s_, k: same_traj_batch(s_, k, 64, 8, 8, n_cross=7))
f0 = jax.jit(lambda s_, k: same_traj_batch(s_, k, 64, 8, 8))
bad_x = 0; pos_x = []; wsum = []; same_prefix = True; n_in = 0; n_tot = 0
for i in range(300):
    b, ix_, iy_, w_, stt = f(st, jax.random.PRNGKey(i)); o = np.asarray(b.obs); w_ = np.asarray(w_); ix_ = np.asarray(ix_); iy_ = np.asarray(iy_)
    b0, ix0, iy0, w0, _ = f0(st, jax.random.PRNGKey(i))
    same_prefix &= np.array_equal(np.asarray(b0.obs), o[:64]) and np.array_equal(np.asarray(ix0), ix_[:448])
    wsum.append((w_[:448].sum(), w_[448:].sum()))
    for g in range(8):
        a = tuple(o[g * 8])
        for k in range(7):
            j = 64 + g * 7 + k; x = tuple(o[j]); n_tot += 1
            pid = 448 + g * 7 + k
            assert ix_[pid] == g * 8 and iy_[pid] == j and ix_[pid + 56] == j and iy_[pid + 56] == g * 8
            if x[0] == a[0] and x[2] == a[2]:
                n_in += 1
                if w_[pid] != 0: bad_x += 1
            else:
                pos_x.append(idxmap[x])
check("mix: st part identical to plain same-traj batch (same key)", same_prefix)
check("mix: weights sum 1 + 1", np.allclose(np.array(wsum), 1.0, atol=1e-5), np.array(wsum).mean(0))
check("mix: in-trajectory cross draws masked", bad_x == 0, (n_in, n_tot))
h = np.histogram(pos_list := pos_x, bins=20, range=(0, 1003))[0]; e_ = len(pos_x) / 20
chi = ((h - e_) ** 2 / e_).sum()
check("mix: cross partners ~uniform over buffer (chi2 df19 < 43.8)", chi < 43.8, round(chi, 1))

# pair-form identity: one group = whole batch, all distinct items
u = jax.random.normal(jax.random.PRNGKey(3), (16,))
a_, b_ = np.meshgrid(np.arange(16), np.arange(16), indexing="ij"); off = a_ != b_
ix, iy = a_[off], b_[off]; w = np.full(len(ix), 1 / len(ix))
lp = 0.25 * float(jnp.sum(w * (u[ix] - u[iy]) ** 2)); lc = 0.5 * 16 / 15 * float(jnp.mean((u - u.mean()) ** 2))
check("pair form == 0.5 m/(m-1) centred loss", abs(lp - lc) < 1e-6, (lp, lc))
print(f"{sum(ok)}/{len(ok)} passed")
