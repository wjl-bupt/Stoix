"""Decision ledger (2026-11-03): what do 1000 real training updates do to the GREEDY decisions, and which training
units (pairs for CW-Pair, samples for DQN) push same-state action rankings the right / wrong way?
usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python decision_ledger.py <state_dir> <cw|dqn> <out_dir> [--check]

* Start from a captured FULL learner state (online, lagged target, Adam moments/count, update counter, replay).
* Replay is FROZEN: T=1000 consecutive real updates with the training loss, optimizer (clip 10 + Adam 5e-4) and hard
  target sync every 1000 updates, exactly as in training, but no new environment data -> isolates the loss.
* Reference = policy-improvement theorem: pi0 = greedy policy before the window; Q^{pi0}_H(s,a) by CRN Monte Carlo
  (64 rollouts/action, H=600, gamma .99, first action forced, then pi0).  Changing the greedy action at s from a0 to a'
  is a one-step improvement iff Q^{pi0}(s,a') > Q^{pi0}(s,a0).
* Probes: 256 states from pi0 (eps=0.1, the training epsilon).  Per probe the decision pair (a0, b*): a0 = greedy
  before, b* = best other action by MC half A (32 rollouts); its label d* = Q^B(s,a0) - Q^B(s,b*) uses the independent
  half B (no selection bias).  sign = sign(d*) if |d*| > 2 SE_B, else undecided.
* Ledger at k = 100/250/500/1000 updates: flips (good / bad / neutral by Q^{pi0} paired 2-SE), non-flip margin moves
  (reinforce / loosen x correct / wrong preference), net one-step improvement dJ = mean_s [Q(s,a_k) - Q(s,a0)].
* Attribution (first order, every update k, 128 probes): Adam update of step k credited to its gradient with the
  momentum carry  coef_k = lr * clip_k * (1 - b1^(T-k+1)) / (1 - b1^c) / (sqrt(nu_hat_k) + eps)  (P_t ~ P_k after k);
  unit j's effect on margin h_s:  iota_{j,s} = (delta_j / n) <grad F_j, coef_k * grad h_s(theta_k)>
  (F_j = D(x,y) for CW, q(s,a) for DQN; the loss gradient is -(1/n) sum_j delta_j grad F_j).  The pre-window momentum
  m_0 is a separate 'history' term.  Alignment of a unit = iota * d*  (>0: moves the margin toward the MC-correct sign).
  Aggregated by unit groups: realised type (diagnostic), pair-state similarity, nearness to the probe (cosine in the
  fixed snapshot-target torso features), whether a unit's action is a0/b* of the probe, |delta| quantile (diagnostic).
Writes <out_dir>/ledger.json and ledger_arrays.npz."""
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, optax, pgx
from jax.flatten_util import ravel_pytree
jax.config.update("jax_compilation_cache_dir", "/home/junweiluo/.cache/jax_ledger")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.ff_rdqn import MinAtarTorso
from stoix.systems.q_learning.dqn_types import Transition
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import DiscreteQNetworkHead

ROOT, ALGO, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
CHECK = "--check" in sys.argv
T_WIN = 30 if CHECK else 1000
G, M, K, LR, AEPS, B1, B2, CLIP, TP = 0.99, 512, 8, 5e-4, 1e-5, 0.9, 0.999, 10.0, 1000
NPROBE, NATT, REPS, H = (32, 16, 16, 100) if CHECK else (256, 128, 64, 600)
CKS = [10, 20, 30] if CHECK else [100, 250, 500, 1000]
clock = time.time()
with open(ROOT / "full_state.pkl", "rb") as f:
    state = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
env = pgx.make("minatar-asterix"); env.sticky_action_prob = 0.1; A = env.num_actions
obs_of = lambda st: st.observation.astype(jnp.float32)
ost = state.opt_states; n0 = int(ost["n"])
exp = jax.tree_util.tree_map(lambda x: x[0], state.buffer_state.experience)
valid_len = int(exp.done.shape[0] if bool(state.buffer_state.is_full) else state.buffer_state.current_index)
exp = Transition(obs=exp.obs, action=exp.action, reward=exp.reward, done=exp.done, next_obs=exp.next_obs, info={})

if ALGO == "cw":
    net = PairNet(num_actions=A, head="free")
    th0, tth0 = state.params.online, state.params.target
    score = lambda th, o: net.apply(th, o, method=PairNet.scores) * (A - 1) / A
    feat = lambda th, o: net.apply(th, o, method=PairNet.phi)
    loss_fn, compute = make_pair_loss(net, G, "cross", 0.33, 1000.0, return_delta=True, n_partners=K)
    def units(tr, sh):
        shifts = 1 + (sh - 1 + jnp.arange(K) * ((M - 1) // K)) % (M - 1)
        ix = jnp.tile(jnp.arange(M), K); iy = ((jnp.arange(M)[None, :] + shifts[:, None]) % M).reshape(-1)
        return ix, iy
    def residual(th, tth, tr, ix, iy):
        return jax.lax.stop_gradient(compute(th, tth, tr, ix, iy)[1]["_delta"])
    def Ffn(th, tr, ix, iy):
        e = net.apply(th, net.apply(th, tr.obs, method=PairNet.phi), tr.action, method=PairNet.sa)
        return net.apply(th, e[ix], e[iy], method=PairNet.D)
    def true_loss(th, tth, tr, sh):
        return loss_fn(th, tth, tr, sh)[0]
else:
    qnet = FeedForwardActor(torso=MinAtarTorso(), action_head=DiscreteQNetworkHead(action_dim=A, epsilon=0.0))
    th0, tth0 = state.params.online["q"], state.params.target["q"]
    assert abs(float(state.params.online["c"])) < 1e-12 and abs(float(state.params.target["c"])) < 1e-12
    score = lambda th, o: qnet.apply(th, o).preferences
    feat = lambda th, o: qnet.apply(th, o, method=lambda mdl, x: mdl.torso(x))
    def units(tr, sh):
        return jnp.arange(M), jnp.arange(M)
    def residual(th, tth, tr, ix, iy):
        r = jnp.clip(tr.reward, -1000.0, 1000.0).astype(jnp.float32)
        y = r + G * (1.0 - tr.done.astype(jnp.float32)) * jnp.max(score(tth, tr.next_obs), -1)
        return jax.lax.stop_gradient(y - score(th, tr.obs)[jnp.arange(M), tr.action])
    def Ffn(th, tr, ix, iy):
        return score(th, tr.obs)[jnp.arange(M), tr.action]
    def true_loss(th, tth, tr, sh):
        r = jnp.clip(tr.reward, -1000.0, 1000.0).astype(jnp.float32)
        y = jax.lax.stop_gradient(r + G * (1.0 - tr.done.astype(jnp.float32)) * jnp.max(score(tth, tr.next_obs), -1))
        return 0.5 * jnp.mean(jnp.square(y - score(th, tr.obs)[jnp.arange(M), tr.action]))
optim = optax.chain(optax.clip_by_global_norm(CLIP), optax.adam(LR, eps=AEPS))
adam0 = ost["q"]
flat0, unravel = ravel_pytree(th0); NP = flat0.shape[0]
print("STATE", ALGO, "n", n0, "adam count", int(adam0[1][0].count), "replay", valid_len, "params", NP,
      "online-target", float(jnp.linalg.norm(flat0 - ravel_pytree(tth0)[0])), flush=True)

# ---------------- probes and MC labels under pi0 ----------------
@jax.jit
def collect(key):
    st = jax.vmap(env.init)(jax.random.split(key, 32))
    def step(st, k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        gr = score(th0, obs_of(st)).argmax(-1)
        act = jnp.where(jax.random.uniform(k1, (32,)) < 0.1, jax.random.randint(k2, (32,), 0, A), gr)
        ns = jax.vmap(env.step)(st, act, jax.random.split(k3, 32))
        fresh = jax.vmap(env.init)(jax.random.split(k4, 32))
        nx = jax.tree_util.tree_map(lambda a, b: jnp.where(ns.terminated.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), fresh, ns)
        return nx, st
    _, traj = jax.lax.scan(step, st, jax.random.split(jax.random.fold_in(key, 1), 400))
    return jax.tree_util.tree_map(lambda x: x[100:].reshape((300 * 32,) + x.shape[2:]), traj)
pool = collect(jax.random.PRNGKey(80001))
pid = jax.random.choice(jax.random.PRNGKey(80002), 9600, (NPROBE,), replace=False)
probe = jax.tree_util.tree_map(lambda x: x[pid], pool); PO = obs_of(probe)

@jax.jit
def mc(sub, basekey):
    n = sub.observation.shape[0]
    st = jax.tree_util.tree_map(lambda x: jnp.repeat(x, A * REPS, axis=0), sub)
    a0 = jnp.tile(jnp.repeat(jnp.arange(A), REPS), n)
    kk = jnp.broadcast_to(jax.random.split(basekey, n * REPS).reshape(n, 1, REPS, 2), (n, A, REPS, 2)).reshape(-1, 2)
    ks = lambda i: jax.vmap(lambda k: jax.random.fold_in(k, i))(kk)
    st = jax.vmap(env.step)(st, a0, ks(0))
    ret = st.rewards[:, 0].astype(jnp.float32); alive = ~st.terminated
    def body(c, i):
        st, ret, alive, disc = c
        ns = jax.vmap(env.step)(st, score(th0, obs_of(st)).argmax(-1), ks(i))
        return (ns, ret + alive * disc * ns.rewards[:, 0], alive & ~ns.terminated, disc * G), None
    (_, ret, alive, _), _ = jax.lax.scan(body, (st, ret, alive, jnp.float32(G)), jnp.arange(1, H))
    return ret.reshape(n, A, REPS), alive.reshape(n, A, REPS)
rets, alv = [], []
CH = 4
for s0 in range(0, NPROBE, CH):
    r, a = mc(jax.tree_util.tree_map(lambda x: x[s0:s0 + CH], probe), jax.random.PRNGKey(90000 + s0))
    rets.append(np.asarray(r)); alv.append(np.asarray(a))
RET = np.concatenate(rets).astype(np.float64); ALIVE = np.concatenate(alv)
t_mc = time.time() - clock
half = REPS // 2
QA, QB, QALL = RET[:, :, :half].mean(-1), RET[:, :, half:].mean(-1), RET.mean(-1)
z0 = np.asarray(score(th0, PO), np.float64); a0 = z0.argmax(-1)
QA_m = QA.copy(); QA_m[np.arange(NPROBE), a0] = -np.inf; bst = QA_m.argmax(-1)
dB = RET[np.arange(NPROBE), a0, half:] - RET[np.arange(NPROBE), bst, half:]
dstar = dB.mean(-1); dse = dB.std(-1, ddof=1) / np.sqrt(REPS - half)
sign = np.where(np.abs(dstar) > 2 * dse, np.sign(dstar), 0.0)
print(f"MC done {t_mc:.0f}s; decided probes {int((sign != 0).sum())}/{NPROBE} (wrong pref {int((sign < 0).sum())}, correct {int((sign > 0).sum())})", flush=True)

a0_j, b_j = jnp.asarray(a0), jnp.asarray(bst)
PO_att = PO[:NATT]; a0_att, b_att = a0_j[:NATT], b_j[:NATT]
dstar_att = jnp.asarray(dstar[:NATT], jnp.float32); sign_att = jnp.asarray(sign[:NATT], jnp.float32)
def hvec(th):
    z = score(th, PO_att)
    return z[jnp.arange(NATT), a0_att] - z[jnp.arange(NATT), b_att]
phi_ref = lambda o: feat(tth0, o)
def unitnorm(x): return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
fP = unitnorm(phi_ref(PO_att))                                             # [NATT, F]

def batch(key):
    k1, k2 = jax.random.split(key)
    ids = jax.random.randint(k1, (M,), 0, valid_len)
    tr = jax.tree_util.tree_map(lambda x: x[ids], exp)
    return tr, jax.random.randint(k2, (), 1, M)

NB = 4
def bins(x, edges): return jnp.clip(jnp.searchsorted(edges, x), 0, NB - 1)
def features(tr, ix, iy, dlt):
    fs = unitnorm(phi_ref(tr.obs))                                         # [M, F]
    d = tr.done.astype(jnp.int32)
    if ALGO == "cw":
        typ = jnp.clip(d[ix] + d[iy], 0, 2)                                # 0 nn, 1 one-terminal, 2 both
        sim = jnp.sum(fs[ix] * fs[iy], -1)
        nearx, neary = fP @ fs[ix].T, fP @ fs[iy].T                        # [NATT, n]
        rel = jnp.maximum(nearx, neary)
        ax, ay = tr.action[ix], tr.action[iy]
        am = ((ax[None] == a0_att[:, None]) | (ax[None] == b_att[:, None]) | (ay[None] == a0_att[:, None]) | (ay[None] == b_att[:, None]))
    else:
        typ = d; sim = jnp.zeros(M)
        rel = fP @ fs.T
        am = (tr.action[None] == a0_att[:, None]) | (tr.action[None] == b_att[:, None])
    return typ, sim, rel, am.astype(jnp.int32), jnp.abs(dlt)

def tangents_jvp(th, tr, ix, iy, V):
    f = lambda v: jax.jvp(lambda t: Ffn(t, tr, ix, iy), (th,), (unravel(v),))[1]
    return jnp.concatenate([jax.vmap(f)(V[i:i + 32]) for i in range(0, V.shape[0], 32)], 0)

@jax.jit
def step(th, tth, opt, key, carry, edges):
    tr, sh = batch(key); ix, iy = units(tr, sh)
    dlt = residual(th, tth, tr, ix, iy); n = dlt.shape[0]
    _, vjpF = jax.vjp(lambda t: Ffn(t, tr, ix, iy), th)
    g = vjpF(-dlt / n)[0]
    gflat = ravel_pytree(g)[0]
    upd, opt2 = optim.update(g, opt, th)
    th2 = optax.apply_updates(th, upd)
    ad = opt2[1][0]; c = ad.count
    nuh = ravel_pytree(ad.nu)[0] / (1 - B2 ** c)
    clipf = jnp.minimum(1.0, CLIP / jnp.maximum(jnp.linalg.norm(gflat), 1e-30))
    coef = LR * clipf * carry / (1 - B1 ** c) / (jnp.sqrt(nuh) + AEPS)
    Jh = jax.jacrev(lambda v: hvec(unravel(v)))(ravel_pytree(th)[0])     # [NATT, NP]
    jv = tangents_jvp(th, tr, ix, iy, Jh * coef[None])                    # [NATT, n]
    iota = jv * (dlt / n)[None]
    typ, sim, rel, am, adl = features(tr, ix, iy, dlt)
    groups = {"type": jnp.broadcast_to(typ[None], iota.shape), "sim": jnp.broadcast_to(bins(sim, edges["sim"])[None], iota.shape),
              "near": bins(rel, edges["near"]), "near_x_actmatch": bins(rel, edges["near"]) * 2 + am,
              "absdelta": jnp.broadcast_to(bins(adl, edges["absdelta"])[None], iota.shape)}
    align = iota * dstar_att[:, None]; sg = iota * sign_att[:, None]
    out = {}
    for gname, L in groups.items():
        nb = 3 if gname == "type" else (2 * NB if gname == "near_x_actmatch" else NB)
        Lf = L.reshape(-1)
        seg = lambda v: jax.ops.segment_sum(v.reshape(-1), Lf, num_segments=nb)
        out[gname] = jnp.stack([seg(align), seg(jnp.abs(iota) * jnp.abs(dstar_att)[:, None]), seg(jnp.maximum(sg, 0)),
                                seg(jnp.maximum(-sg, 0)), seg(jnp.abs(iota) * (sign_att != 0)[:, None]), seg(jnp.ones_like(iota))])
    pred = iota.sum(1)
    return th2, opt2, out, pred, hvec(th2) - hvec(th), jnp.linalg.norm(gflat) > CLIP

# edges from a first batch (fixed for the whole window)
tr_e, sh_e = batch(jax.random.PRNGKey(7))
ix_e, iy_e = units(tr_e, sh_e)
_, sim_e, rel_e, _, adl_e = features(tr_e, ix_e, iy_e, residual(th0, tth0, tr_e, ix_e, iy_e))
q3 = jnp.array([.25, .5, .75])
edges = {"sim": jnp.quantile(sim_e, q3) if ALGO == "cw" else jnp.array([0.0, 0.0, 0.0]), "near": jnp.quantile(rel_e, q3),
         "absdelta": jnp.quantile(adl_e, q3)}
# exactness checks of the decomposition on one batch
gl = ravel_pytree(jax.grad(lambda t: true_loss(t, tth0, tr_e, sh_e))(th0))[0]
dl = residual(th0, tth0, tr_e, ix_e, iy_e); _, vj = jax.vjp(lambda t: Ffn(t, tr_e, ix_e, iy_e), th0)
gm = ravel_pytree(vj(-dl / dl.shape[0])[0])[0]
grad_check = float(jnp.linalg.norm(gm - gl) / jnp.linalg.norm(gl))
print("grad split check (unit decomposition vs autodiff loss)", grad_check, flush=True)

# history (pre-window momentum) term, at theta_0 with nu_hat of the snapshot
ad0 = adam0[1][0]; c0 = ad0.count
m0 = ravel_pytree(ad0.mu)[0]; nuh0 = ravel_pytree(ad0.nu)[0] / (1 - B2 ** c0)
hist_dtheta = -LR * (B1 * (1 - B1 ** T_WIN) / (1 - B1)) * m0 / (1 - B1 ** (c0 + 1)) / (jnp.sqrt(nuh0) + AEPS)
Jh0 = jax.jacrev(lambda v: hvec(unravel(v)))(flat0)
hist_pred = np.asarray(Jh0 @ hist_dtheta)

th, tth, opt = th0, tth0, adam0
acc = None; pred_sum = np.zeros(NATT); act_sum = np.zeros(NATT); nclip = 0; ledger = {}
zs = {0: z0}
keys = jax.random.split(jax.random.PRNGKey(424242), T_WIN)
for k in range(T_WIN):
    carry = jnp.float32((1 - B1 ** (T_WIN - k)) / (1 - B1) * (1 - B1))   # (1-b1) * sum_{t>=k} b1^(t-k), truncated at T
    th, opt, out, pred, dh, clipped = step(th, tth, opt, keys[k], carry, edges)
    out = jax.tree_util.tree_map(np.asarray, out)
    acc = out if acc is None else jax.tree_util.tree_map(lambda a, b: a + b, acc, out)
    pred_sum += np.asarray(pred); act_sum += np.asarray(dh); nclip += int(clipped)
    if (n0 + k + 1) % TP == 0:
        tth = th
        print("target sync at window step", k + 1, flush=True)
    if (k + 1) in CKS:
        zs[k + 1] = np.asarray(score(th, PO), np.float64)
        print(f"step {k+1}  {time.time()-clock:.0f}s", flush=True)

# ---------------- ledger ----------------
def paired(a_new):
    dd = RET[np.arange(NPROBE), a_new] - RET[np.arange(NPROBE), a0]
    return dd.mean(-1), dd.std(-1, ddof=1) / np.sqrt(REPS)
for kk, z in zs.items():
    if kk == 0: continue
    ak = z.argmax(-1); flip = ak != a0
    dq, dqse = paired(ak)
    m0_ = z0[np.arange(NPROBE), a0] - z0[np.arange(NPROBE), bst]; mk = z[np.arange(NPROBE), a0] - z[np.arange(NPROBE), bst]
    dm = mk - m0_
    nf = ~flip
    L = {"flip_good": int((flip & (dq > 2 * dqse)).sum()), "flip_bad": int((flip & (dq < -2 * dqse)).sum()),
         "flip_neutral": int((flip & (np.abs(dq) <= 2 * dqse)).sum()),
         "reinforce_correct": int((nf & (sign > 0) & (dm > 0)).sum()), "loosen_correct": int((nf & (sign > 0) & (dm < 0)).sum()),
         "reinforce_wrong": int((nf & (sign < 0) & (dm > 0)).sum()), "loosen_wrong": int((nf & (sign < 0) & (dm < 0)).sum()),
         "undecided_nonflip": int((nf & (sign == 0)).sum()),
         "dJ_mean": float(dq.mean()), "dJ_se": float(dq.std(ddof=1) / np.sqrt(NPROBE)),
         "dJ_flipped_only_sum_over_probes": float(dq[flip].sum()),
         "margin_alignment_mean": float(np.mean(dm * dstar)), "margin_alignment_se": float(np.std(dm * dstar, ddof=1) / np.sqrt(NPROBE)),
         "frac_flipped": float(flip.mean())}
    ledger[str(kk)] = L
# ---------------- attribution summary ----------------
def summ(a):
    al, nrm, pos, neg, mass, cnt = a
    return {"alignment_sum": al.tolist(), "alignment_ratio": (al / np.maximum(nrm, 1e-30)).tolist(),
            "good_frac_decided": (pos / np.maximum(pos + neg, 1e-30)).tolist(), "abs_mass_decided": mass.tolist(),
            "n_unit_probe": cnt.tolist()}
attr = {g: summ(v) for g, v in acc.items()}
tot = acc["type"]
lin_err = float(np.linalg.norm(pred_sum + hist_pred - (zs[T_WIN][np.arange(NPROBE), a0] - zs[T_WIN][np.arange(NPROBE), bst]
                                                         - (z0[np.arange(NPROBE), a0] - z0[np.arange(NPROBE), bst]))[:NATT])
                / max(np.linalg.norm(act_sum), 1e-30))
res = {"algo": ALGO, "state_dir": str(ROOT), "T_window": T_WIN, "n_probe": NPROBE, "n_attr_probe": NATT, "reps": REPS, "H": H,
       "update_counter_start": n0, "replay_items": valid_len, "n_params": int(NP), "grad_split_check": grad_check,
       "clip_steps": nclip, "decided_probes": int((sign != 0).sum()), "wrong_pref_probes": int((sign < 0).sum()),
       "correct_pref_probes": int((sign > 0).sum()), "mc_alive_frac_at_H": float(ALIVE.mean()),
       "ledger": ledger, "attribution": attr,
       "attribution_total_alignment": float(tot[0].sum()), "attribution_total_ratio": float(tot[0].sum() / max(tot[1].sum(), 1e-30)),
       "history_alignment": float(np.sum(hist_pred * dstar[:NATT])),
       "history_ratio": float(np.sum(hist_pred * dstar[:NATT]) / max(np.sum(np.abs(hist_pred) * np.abs(dstar[:NATT])), 1e-30)),
       "actual_margin_change_alignment_attprobes": float(np.sum(act_sum * dstar[:NATT])),
       "first_order_vs_actual_rel_err": lin_err,
       "corr_pred_vs_actual": float(np.corrcoef(pred_sum + hist_pred, act_sum)[0, 1]),
       "edges": {k: np.asarray(v).tolist() for k, v in edges.items()}, "elapsed": time.time() - clock}
(OUT / "ledger.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / "ledger_arrays.npz", RET=RET.astype(np.float32), z=np.stack([zs[k] for k in sorted(zs)]),
                    ck=np.array(sorted(zs)), a0=a0, bst=bst, dstar=dstar, dse=dse, pred_sum=pred_sum, hist_pred=hist_pred, act_sum=act_sum)
print(json.dumps({k: res[k] for k in ("ledger", "attribution_total_ratio", "history_ratio", "first_order_vs_actual_rel_err", "corr_pred_vs_actual", "grad_split_check", "elapsed")}), flush=True)
sys.stdout.flush(); os._exit(0)
