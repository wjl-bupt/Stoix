"""Checks for mujoco_offpolicy.py (2026-11-07).  Run: JAX_PLATFORMS=cpu python check_mujoco.py"""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, ".")
import sys as _s; _s.path.insert(0, "/home/junweiluo/Stoix/cwq/mujoco_gym")
import mujoco_offpolicy as M

ok = []
def check(name, cond):
    ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)

rng = np.random.default_rng(0)
m, g = 64, 0.99
# 1. cross_xi with target = online  ==  v_y u_x - v_x u_y
q = rng.normal(size=m); qn = rng.normal(size=m); r = rng.normal(size=m); d = (rng.random(m) < .2).astype(float)
D = q[:, None] - q[None]; Mm = qn[:, None] - qn[None]; Amx = qn[:, None] - q[None]; Bmn = q[:, None] - qn[None]
xi = np.asarray(M.cross_xi(D, Mm, Amx, Bmn, r[:, None], r[None], d[:, None], d[None], g))
G = g * (1 - d); v = 1 - G; u = r + G * qn - q
check("xi == v_y u_x - v_x u_y (target=online)", np.allclose(xi, v[None] * u[:, None] - v[:, None] * u[None], atol=1e-9))

# 2. critic loss: qcw with no termination, uniform W, target=online  ==  0.5 * sum W (u_x - u_y)^2  (centered TD)
qnet = M.QNet(); od, ad = 5, 2
p = qnet.init(jax.random.PRNGKey(0), jnp.zeros((1, od)), jnp.zeros((1, ad)))
p2 = qnet.init(jax.random.PRNGKey(1), jnp.zeros((1, od)), jnp.zeros((1, ad)))
b = {"obs": jnp.asarray(rng.normal(size=(m, od)), jnp.float32), "action": jnp.asarray(rng.uniform(-1, 1, (m, ad)), jnp.float32),
     "r": jnp.asarray(rng.normal(size=m), jnp.float32), "d": jnp.zeros(m, jnp.float32),
     "next_obs": jnp.asarray(rng.normal(size=(m, od)), jnp.float32)}
a_n = jnp.asarray(rng.uniform(-1, 1, (m, ad)), jnp.float32)
W = jnp.asarray((1 - np.eye(m)) / (m * (m - 1)), jnp.float32)
L, info = M.critic_loss_fn(qnet, (p,), (p,), b, a_n, g, "qcw", W)
qo = qnet.apply(p, b["obs"], b["action"]); qn_ = qnet.apply(p, b["next_obs"], a_n)
uu = np.asarray(b["r"] + g * qn_ - qo)
ref = 0.5 * np.sum(np.asarray(W) * (uu[:, None] - uu[None]) ** 2)
check("qcw loss == centered TD (d=0, uniform W)", np.isclose(float(L), ref, rtol=1e-4))
check("centered TD == m/(m-1) var(u)", np.isclose(ref, m / (m - 1) * np.var(uu), rtol=1e-4))
# gradient of qcw with level shift invariance: adding constant to Q leaves loss unchanged when d=0
# (checked through u: u - c for all items -> differences unchanged) -> trivially true; check twin min logic
L2, info2 = M.critic_loss_fn(qnet, (p, p2), (p, p2), b, a_n, g, "q", W)
qa = np.asarray(qnet.apply(p, b["next_obs"], a_n)); qb = np.asarray(qnet.apply(p2, b["next_obs"], a_n))
check("pick0 diagnostic", np.isclose(float(info2["pick0"]), np.mean(qa <= qb)))
y = np.asarray(b["r"]) + g * np.minimum(qa, qb)
ref2 = sum(0.5 * np.mean((y - np.asarray(qnet.apply(pp, b["obs"], b["action"]))) ** 2) for pp in (p, p2))
check("TD3 critic loss == sum_k 0.5 mse(y_min)", np.isclose(float(L2), ref2, rtol=1e-5))
# terminal: d=1 removes bootstrap in q head
b1 = dict(b); b1["d"] = jnp.ones(m, jnp.float32)
L3, _ = M.critic_loss_fn(qnet, (p,), (p,), b1, a_n, g, "q", W)
check("d=1 -> y = r", np.isclose(float(L3), 0.5 * np.mean((np.asarray(b["r"]) - np.asarray(qo)) ** 2), rtol=1e-5))

# 3. rho_weight_matrix: sums, zero for identical items
idx = jnp.asarray(np.array([0, 1, 1, 2, 5, 6, 7, 7]))
Wr = np.asarray(M.rho_weight_matrix(idx, 4, 0.5))
grp = np.arange(8) // 4
same = (grp[:, None] == grp[None]); dist = np.asarray(idx)[:, None] != np.asarray(idx)[None]
check("rho W: same-group sum .5", np.isclose(Wr[same].sum(), .5))
check("rho W: cross-group sum .5", np.isclose(Wr[~same].sum(), .5))
check("rho W: identical items weight 0", np.all(Wr[~dist] == 0))

# 4. replay rho sampler: partners in the anchor's trajectory, never the anchor itself unless singleton
R = M.Replay(5000, 3, 1); ends = set()
t = 0
while R.n < 4000:
    L_ep = int(rng.integers(1, 60))
    for j in range(L_ep):
        if R.n >= 4000: break
        R.add(np.full(3, R.n), np.zeros(1), 0.0, np.full(3, R.n + 1), 0.0, j == L_ep - 1)
ep = R.ep[:R.n]
good_traj, good_self, cnt_single = True, True, 0
for _ in range(200):
    bt, idx = R.sample_rho(rng, 256, 8)
    gi = idx.reshape(32, 8)
    for row in gi:
        a = row[0]
        if np.sum(ep == ep[a]) == 1:
            cnt_single += 1; good_self &= np.all(row == a)
        else:
            good_traj &= np.all(ep[row] == ep[a]); good_self &= np.all(row[1:] != a)
    check_obs = np.all(bt["obs"][:, 0] == idx)
check("rho sampler: partners in anchor trajectory", good_traj)
check("rho sampler: partner != anchor (non-singleton)", good_self)
check("rho sampler: obs gathered at idx", check_obs)
# uniform marginal of partners within the trajectory: chi-like check on a long trajectory
R2 = M.Replay(2000, 1, 1)
for i in range(1000): R2.add(np.zeros(1), np.zeros(1), 0., np.zeros(1), 0., i == 999)
cnt = np.zeros(1000)
for _ in range(400):
    _, idx = R2.sample_rho(rng, 256, 8); cnt += np.bincount(idx, minlength=1000)
check("rho sampler: slots ~ uniform over one trajectory (cv<.1)", cnt.std() / cnt.mean() < 0.1)

# 5. ablation heads (2026-11-07)
p_t = qnet.init(jax.random.PRNGKey(7), jnp.zeros((1, od)), jnp.zeros((1, ad)))       # target != online
Lcw, _ = M.critic_loss_fn(qnet, (p,), (p_t,), b, a_n, g, "qcw", W)
Lct, _ = M.critic_loss_fn(qnet, (p,), (p_t,), b, a_n, g, "qct", W)
check("qct == qcw when d=0 (target != online)", np.isclose(float(Lcw), float(Lct), rtol=1e-4))
bt = dict(b); bt["d"] = jnp.asarray((rng.random(m) < .2).astype(np.float32))
Lcw2, _ = M.critic_loss_fn(qnet, (p,), (p_t,), bt, a_n, g, "qcw", W)
Lct2, _ = M.critic_loss_fn(qnet, (p,), (p_t,), bt, a_n, g, "qct", W)
qo = np.asarray(qnet.apply(p, bt["obs"], bt["action"])); qn_t = np.asarray(qnet.apply(p_t, bt["next_obs"], a_n))
uu = np.asarray(bt["r"]) + g * (1 - np.asarray(bt["d"])) * qn_t - qo
check("qct == 0.5 sum W (u_x-u_y)^2 with terminals", np.isclose(float(Lct2), 0.5 * np.sum(np.asarray(W) * (uu[:, None] - uu[None]) ** 2), rtol=1e-4))
check("qct != qcw with terminals", not np.isclose(float(Lcw2), float(Lct2), rtol=1e-3))
import copy
p_s = jax.tree_util.tree_map(lambda x: x, p)
p_s = {"params": dict(p["params"])}; p_s["params"]["Dense_2"] = dict(p["params"]["Dense_2"]); p_s["params"]["Dense_2"]["bias"] = p["params"]["Dense_2"]["bias"] + 5.0
q0n = np.asarray(qnet.apply(p, b["next_obs"], a_n))
Lc, infc = M.critic_loss_fn(qnet, (p, p_s), (p, p_s), b, a_n, g, "q", W, "centered")
q0o = np.asarray(qnet.apply(p, b["obs"], b["action"])); rr = np.asarray(b["r"])
ref_c = 0.5 * np.mean((rr + g * q0n - q0o) ** 2) + 0.5 * np.mean((rr + g * (q0n + 5) - (q0o + 5)) ** 2)
check("centered min: level-shifted twin -> each critic its own level", np.isclose(float(Lc), ref_c, rtol=1e-4))
Lr, infr = M.critic_loss_fn(qnet, (p, p_s), (p, p_s), b, a_n, g, "q", W, "raw")
ref_r = 0.5 * np.mean((rr + g * q0n - q0o) ** 2) + 0.5 * np.mean((rr + g * q0n - (q0o + 5)) ** 2)
check("raw min: shifted twin hijacked (pick0=1, pick0c=1 tie)", np.isclose(float(Lr), ref_r, rtol=1e-4) and float(infr["pick0"]) == 1.0)
Lcc, _ = M.critic_loss_fn(qnet, (p, p2), (p, p2), b, a_n, g, "qcw", W, "centered")
check("centered min runs with qcw (finite)", np.isfinite(float(Lcc)))

Ln, infn = M.critic_loss_fn(qnet, (p, p2), (p, p2), b, a_n, g, "qcw", W, "none")
L1a, _ = M.critic_loss_fn(qnet, (p,), (p,), b, a_n, g, "qcw", W)
L1b, _ = M.critic_loss_fn(qnet, (p2,), (p2,), b, a_n, g, "qcw", W)
check("min_mode none == sum of independent single-critic losses", np.isclose(float(Ln), float(L1a) + float(L1b), rtol=1e-5))
print(f"{sum(ok)}/{len(ok)} passed")
