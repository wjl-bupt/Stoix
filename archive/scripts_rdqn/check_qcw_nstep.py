"""2026-11-09 checks for n-step CW-Q-rho (same_traj.nstep_tail + duel_all_pairs g_item)."""
import json, pickle, sys
from types import SimpleNamespace
from typing import NamedTuple
class Exp(NamedTuple):
    reward: object
    done: object
    next_obs: object
import numpy as np
sys.path.insert(0, "/home/junweiluo/Stoix")
import jax, jax.numpy as jnp
from stoix.systems.q_learning.same_traj import nstep_tail, rho_weight_matrix
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
# ---- T1 tail vs brute force on a wrapped ring buffer ----
g = 0.99
for (E, L, W, nst, seed) in [(3, 24, 40, 3, 0), (4, 40, 33, 3, 1), (5, 50, 200, 4, 2), (3, 30, 30, 3, 3)]:
    rng = np.random.default_rng(seed)
    num = np.full(L, -1); rew = np.zeros(L, np.float32); don = np.zeros(L, bool); nxo = np.zeros((L, 2), np.float32)
    rw_all = rng.normal(size=W).astype(np.float32); dn_all = rng.random(W) < .2
    for w in range(W):
        pos = w % L; num[pos] = w; rew[pos] = rw_all[w]; don[pos] = dn_all[w]; nxo[pos] = [w, -w]
    bs = SimpleNamespace(experience=Exp(reward=jnp.asarray(rew)[None], done=jnp.asarray(don)[None], next_obs=jnp.asarray(nxo)[None]),
                         current_index=jnp.asarray(W % L), is_full=jnp.asarray(W >= L))
    valid = [pos for pos in range(L) if num[pos] >= 0]
    idx = jnp.asarray(valid)
    R, Gm, last, nu = map(np.asarray, nstep_tail(bs, idx, E, nst, g, 1000.0))
    good = True
    for t, pos in enumerate(valid):
        w = num[pos]; Rm = 0.0; Gb = None; lw = w; used = 0
        for k in range(nst):
            wk = w + k * E
            if wk >= W: break
            Rm += g ** k * rw_all[wk]; used += 1; lw = wk
            if dn_all[wk]: Gb = 0.0; break
            Gb = g ** (k + 1)
        good &= abs(R[t] - Rm) < 1e-5 and abs(Gm[t] - Gb) < 1e-6 and nxo[last[t]][0] == lw and nu[t] == used
    chk(good, f"nstep_tail == brute force (E={E}, L={L}, writes={W}, n={nst}, {len(valid)} items)")
# ---- T2/T3 loss with per-item Gam ----
R0 = "/home/junweiluo/stoix_runs/minatar_g8ck/"
net = PairNet(num_actions=3, head="qcw", hidden=128, emb_dim=16, rank=0, n_atoms=51, vmax=20.0, additive=False)
ld = lambda f: jax.tree_util.tree_map(lambda x: x[0], pickle.load(open(R0 + "ckpt/" + f, "rb")))
p, tp = ld("breakout_qcw_rho_g8ck_s0_t4998400.pkl"), ld("breakout_qcw_rho_g8ck_s0_t3998720.pkl")
O = np.asarray(pickle.load(open("/home/junweiluo/stoix_runs/breakout_miss/bank.pkl", "rb")).observation)
rng = np.random.default_rng(3); m, nst = 256, 3
oi = rng.integers(0, len(O), m); a = rng.integers(0, 3, m); ni = rng.integers(0, len(O), m)
dwin = rng.random(m) < .1; short = rng.random(m) < .1
Gam = np.where(dwin, 0.0, np.where(short, g ** 2, g ** nst)).astype(np.float32)
Rn = rng.normal(size=m).astype(np.float32) * .3
tr = SimpleNamespace(obs=jnp.asarray(O[oi]), next_obs=jnp.asarray(O[ni]), action=jnp.asarray(a), reward=jnp.asarray(Rn),
                     done=jnp.asarray(dwin), info={"nstep_G": jnp.asarray(Gam)})
pw = rho_weight_matrix(jnp.arange(m), 8, 0.5); ix = jnp.arange(m)
_, comp = make_pair_loss(net, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True, n_step=nst)
flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
gr, info = jax.grad(lambda pp: comp(pp, p, tr, ix, ix, pw=pw), has_aux=True)(p)      # target = online
Qf = lambda pp, o: net.apply(pp, o, method=PairNet.duel_comp_obs)[3]
Qi = Qf(p, tr.obs)[jnp.arange(m), tr.action]; Vn = Qf(p, tr.next_obs).max(-1)
u = tr.reward + Gam * Vn - Qi; v = 1 - Gam
Ws = pw + pw.T
c = -((Ws @ v) * u - (Ws @ u) * v) / (1 - g ** nst)
gt = jax.grad(lambda pp: jnp.sum(jax.lax.stop_gradient(c) * Qf(pp, tr.obs)[jnp.arange(m), tr.action]))(p)
d = float(jnp.max(jnp.abs(flat(gr) - flat(gt))) / jnp.max(jnp.abs(flat(gt))))
chk(d < 1e-4, f"target=online: gradient == theory xi = v_j u_i - v_i u_j with n-step u, v=1-Gam (max rel diff {d:.1e})")
gr2, _ = jax.grad(lambda pp: comp(pp, tp, tr, ix, ix, pw=pw), has_aux=True)(p)
bs_ = float(jnp.abs(gr2["params"]["du_adv"]["bias"].sum()) / jnp.max(jnp.abs(flat(gr2))))
chk(bs_ < 1e-5, f"gauge-neutral with lagged target (|sum bias grad|/max {bs_:.1e})")
# n_step=1 code path untouched: make_pair_loss(n_step=1) without nstep_G
tr1 = SimpleNamespace(obs=tr.obs, next_obs=tr.next_obs, action=tr.action, reward=tr.reward, done=tr.done, info={})
_, comp1 = make_pair_loss(net, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True)
l1, _ = comp1(p, tp, tr1, ix, ix, pw=pw)
# same as n-step code with Gam = g(1-d), n=1 formula
tr1b = SimpleNamespace(obs=tr.obs, next_obs=tr.next_obs, action=tr.action, reward=tr.reward, done=tr.done,
                       info={"nstep_G": g * (1 - tr.done.astype(jnp.float32))})
_, comp1b = make_pair_loss(net, g, "cross", 0.33, 1000.0, all_pairs=True, cur_target=True, n_step=1)
l1b, _ = comp1b(p, tp, tr1b, ix, ix, pw=pw)
chk(float(l1) == float(l1b), f"n_step=1 ignores nstep_G / unchanged ({float(l1):.6f})")
print(f"{ok}/{n} passed")
