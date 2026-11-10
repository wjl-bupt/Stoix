"""DIAG 2026-11-02: is the variance of terminal pairs harmful?  Gradient attribution by pair type on checkpoints.
usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python grad_attr.py <run_dir> <game> <algo> <seed> <t> [--check]
algo in {cwpair_mlp, dqn}.  Checkpoints = online params from diag_gap (no target net, no replay saved).

1. Replay proxy: 128 envs play the checkpoint's own eps=0.1 policy (training epsilon), 300 burn-in steps, then 800
   steps -> 102,400 transitions (~ replay capacity 100k).  next_obs is the pre-reset observation.
2. B=256 minibatches of m=512 (uniform with replacement), K=8 cyclic-shift partners exactly as loss_fn.
   Target network := online params (approximation; target period is only 1000 updates).
3. Per minibatch the gradient of the training loss is split exactly by pair type:
     CW-Pair: g = g_nn + g_one + g_both     (d_x=d_y=0 / exactly one terminal / both terminal),
              g_T = -(1/n_pairs) sum_{j in T} delta_j grad D_j     (same normalisation as the loss)
     DQN:     g = g_nonterm + g_term,  g_T = -(1/m) sum_{i in T} delta_i grad q_i
4. Over the B minibatches (gbar = mean, Var = sum of per-coordinate variances):
     signal share  s_T = <gbar_T, gbar> / |gbar|^2
     noise share   N_T / N,  N = sum_i Var(g_i)
     SNR = |gbar|^2 / N ;  counterfactual SNR_cf: terminal-type gradients replaced by their mean (keep the signal,
          remove only their sampling noise): g' = g_rest + gbar_term  =>  N_cf = sum_i Var(g_rest,i)
     Adam proxy  P = sum_i gbar_i^2 / sqrt(E g_i^2)   (progress of one Adam-like step along the mean gradient);
          P_cf uses E g'_i^2 = Var(g_rest,i) + gbar_i^2;  report P_cf / P
     clip fraction: P(|g|_2 > 10) (max_grad_norm of the protocol)
   "terminal types" = one+both for CW-Pair, term for DQN.
Pre-registered reading: P_cf/P >= 2 (or SNR_cf/SNR >= 2) -> terminal-pair variance is a material cost worth a
variance-only intervention; <= 1.3 -> not a bottleneck.  Writes <run_dir>/gradattr/<game>_<algo>_s<seed>_t<t>.json"""
import json, os, pickle, sys, time
import numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
jax.config.update("jax_compilation_cache_dir", "/home/junweiluo/.cache/jax_gradattr")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
import pgx
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rdqn import MinAtarTorso
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.dqn_types import Transition
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import DiscreteQNetworkHead

run, game, algo, seed, t = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
CHECK = "--check" in sys.argv
G, EPS, NENV, BURN, T, M, K, B, MAXR, CLIP = 0.99, 0.1, 128, 300, 800, 512, 8, 256, 1000.0, 10.0
env = pgx.make(f"minatar-{game}"); env.sticky_action_prob = 0.1
A = env.num_actions
with open(f"{run}/ckpt/{game}_{algo}_s{seed}_t{t}.pkl", "rb") as f:
    p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
obs_of = lambda st: st.observation.astype(jnp.float32)

if algo == "dqn":
    dnet = FeedForwardActor(torso=MinAtarTorso(), action_head=DiscreteQNetworkHead(action_dim=A, epsilon=0.0))
    qall = lambda pp, o: dnet.apply(pp["q"], o).preferences
    score = qall
else:
    net = PairNet(num_actions=A, head="free")
    score = lambda pp, o: net.apply(pp, o, method=PairNet.scores)
    loss_fn, compute = make_pair_loss(net, G, "cross", 0.33, MAXR, return_delta=True, n_partners=K)

# ---------------- 1. replay proxy ----------------
def collect(pp, key):
    k0, key = jax.random.split(key)
    st = jax.vmap(env.init)(jax.random.split(k0, NENV))
    def body(st, k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        o = obs_of(st)
        g = jnp.argmax(score(pp, o), -1)
        act = jnp.where(jax.random.uniform(k2, g.shape) < EPS, jax.random.randint(k1, g.shape, 0, A), g)
        nst = jax.vmap(env.step)(st, act, jax.random.split(k3, NENV))
        rec = (o, act, nst.rewards[:, 0].astype(jnp.float32), nst.terminated, obs_of(nst))
        fresh = jax.vmap(env.init)(jax.random.split(k4, NENV))
        nst = jax.tree_util.tree_map(
            lambda f_, s_: jnp.where(nst.terminated.reshape((-1,) + (1,) * (s_.ndim - 1)), f_, s_), fresh, nst)
        return nst, rec
    _, traj = jax.lax.scan(body, st, jax.random.split(key, BURN + T))
    return jax.tree_util.tree_map(lambda x: x[BURN:].reshape((T * NENV,) + x.shape[2:]), traj)

t0 = time.time()
O, ACT, R, D, NO = jax.jit(collect)(p, jax.random.PRNGKey(7919 * seed + t % 10007))
N = O.shape[0]
term_frac = float(D.mean())

# ---------------- 2./3. per-type gradients ----------------
flat0, unravel = ravel_pytree(p)
nparam = flat0.shape[0]

def batch(key):
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (M,), 0, N)
    tr = Transition(obs=O[idx], action=ACT[idx], reward=R[idx], done=D[idx], next_obs=NO[idx], info={})
    shift = jax.random.randint(k2, (), 1, M)
    return tr, shift

if algo == "dqn":
    def type_grads(pp, tr, shift):
        qn = jax.lax.stop_gradient(qall(pp, tr.next_obs)).max(-1)
        d = tr.done.astype(jnp.float32)
        y = jnp.clip(tr.reward, -MAXR, MAXR) + G * (1 - d) * qn
        qa_fn = lambda q: qall(q, tr.obs)[jnp.arange(M), tr.action]
        qa, vjp = jax.vjp(qa_fn, pp)
        delta = jax.lax.stop_gradient(y - qa)
        gs = [ravel_pytree(vjp(-delta * w / M)[0])[0] for w in (1 - d, d)]
        full = 0.5 * jnp.mean(jnp.square(y - qa))
        return jnp.stack(gs), jnp.array([jnp.mean(1 - d), jnp.mean(d)]), delta
    TYPES, TERM = ["nonterm", "term"], [1]
else:
    def type_grads(pp, tr, shift):
        shifts = 1 + (shift - 1 + jnp.arange(K) * ((M - 1) // K)) % (M - 1)
        ix = jnp.tile(jnp.arange(M), K)
        iy = ((jnp.arange(M)[None, :] + shifts[:, None]) % M).reshape(-1)
        _, info = compute(pp, pp, tr, ix, iy)
        delta = jax.lax.stop_gradient(info["_delta"])
        def Dfn(q):
            e = net.apply(q, net.apply(q, tr.obs, method=PairNet.phi), tr.action, method=PairNet.sa)
            return net.apply(q, e[ix], e[iy], method=PairNet.D)
        _, vjp = jax.vjp(Dfn, pp)
        d = tr.done.astype(jnp.float32); dx, dy = d[ix], d[iy]
        w_nn = (1 - dx) * (1 - dy); w_both = dx * dy; w_one = 1 - w_nn - w_both
        npair = ix.shape[0]
        gs = [ravel_pytree(vjp(-delta * w / npair)[0])[0] for w in (w_nn, w_one, w_both)]
        return jnp.stack(gs), jnp.array([w_nn.mean(), w_one.mean(), w_both.mean()]), delta
    TYPES, TERM = ["nn", "one", "both"], [1, 2]

type_grads_j = jax.jit(type_grads)

if CHECK:   # the per-type split must sum to the true training-loss gradient
    tr, sh = batch(jax.random.PRNGKey(1))
    gs, _, _ = type_grads_j(p, tr, sh)
    if algo == "dqn":
        def L(q):
            qn = jax.lax.stop_gradient(qall(p, tr.next_obs)).max(-1); d = tr.done.astype(jnp.float32)
            y = tr.reward + G * (1 - d) * qn
            return 0.5 * jnp.mean(jnp.square(y - qall(q, tr.obs)[jnp.arange(M), tr.action]))
        gt = ravel_pytree(jax.grad(L)(p))[0]
    else:
        gt = ravel_pytree(jax.grad(lambda q: loss_fn(q, p, tr, sh)[0])(p))[0]
    err = float(jnp.abs(gs.sum(0) - gt).max() / (jnp.abs(gt).max() + 1e-12))
    errn = float(jnp.linalg.norm(gs.sum(0) - gt) / (jnp.linalg.norm(gt) + 1e-12))
    print(f"CHECK sum of type gradients vs autodiff: rel max err {err:.2e}, rel norm err {errn:.2e}",
          "PASS" if errn < 1e-3 else "FAIL")
    if algo != "dqn":   # float32 floor: the same autodiff gradient with pairs summed in a different order
        shifts = 1 + (sh - 1 + jnp.arange(K) * ((M - 1) // K)) % (M - 1)
        perm = jnp.argsort(jax.random.uniform(jax.random.PRNGKey(3), (K,)))
        ix = jnp.tile(jnp.arange(M), K); iy = ((jnp.arange(M)[None, :] + shifts[perm][:, None]) % M).reshape(-1)
        gt3 = ravel_pytree(jax.grad(lambda q: compute(q, p, tr, ix, iy)[0])(p))[0]
        print(f"CHECK float32 floor (same loss, partner blocks reordered): rel norm {float(jnp.linalg.norm(gt3 - gt) / jnp.linalg.norm(gt)):.2e}")

@jax.jit
def accumulate(acc, key):
    tr, sh = batch(key)
    gs, fr, delta = type_grads_j(p, tr, sh)
    g = gs.sum(0)
    gt = gs[jnp.array(TERM)].sum(0)
    rest = g - gt
    S, S2, Sr, Sr2, St, St2, nrm, frs, dsq = acc
    return (S + gs, S2 + gs ** 2, Sr + jnp.stack([g, rest]), Sr2 + jnp.stack([g, rest]) ** 2, St + gt, St2 + gt ** 2,
            nrm + jnp.array([jnp.linalg.norm(g) > CLIP, jnp.linalg.norm(g)]), frs + fr,
            dsq + jnp.array([jnp.mean(delta ** 2)])), None

nt = len(TYPES)
acc0 = (jnp.zeros((nt, nparam)), jnp.zeros((nt, nparam)), jnp.zeros((2, nparam)), jnp.zeros((2, nparam)),
        jnp.zeros(nparam), jnp.zeros(nparam), jnp.zeros(2), jnp.zeros(nt), jnp.zeros(1))
acc = acc0
keys = jax.random.split(jax.random.PRNGKey(31 * seed + 5), B)
for k in keys:
    acc, _ = accumulate(acc, k)
S, S2, Sr, Sr2, St, St2, nrm, frs, dsq = [np.asarray(x, np.float64) for x in acc]
mean = lambda s: s / B
var = lambda s, s2: np.maximum(s2 / B - (s / B) ** 2, 0) * B / (B - 1)
gbar_T = mean(S); gbar = mean(Sr[0]); gbar_rest = mean(Sr[1]); gbar_term = mean(St)
N_full = var(Sr[0], Sr2[0]).sum(); N_rest = var(Sr[1], Sr2[1]).sum(); N_term = var(St, St2).sum()
N_T = var(S, S2).sum(-1)
g2 = gbar @ gbar
Eg2 = var(Sr[0], Sr2[0]) + gbar ** 2
Eg2_cf = var(Sr[1], Sr2[1]) + gbar ** 2
P_adam = float(np.sum(gbar ** 2 / np.sqrt(Eg2 + 1e-30)))
P_cf = float(np.sum(gbar ** 2 / np.sqrt(Eg2_cf + 1e-30)))
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
out = {"game": game, "algo": algo, "seed": seed, "t": t, "n_pool": int(N), "term_frac_pool": term_frac,
       "types": TYPES, "pair_frac": (frs / B).tolist(), "mean_delta_sq": float(dsq[0] / B),
       "signal_share": {TYPES[i]: float(gbar_T[i] @ gbar / g2) for i in range(nt)},
       "noise_share": {TYPES[i]: float(N_T[i] / N_full) for i in range(nt)},
       "noise_share_terminal_types": float(N_term / N_full),
       "signal_share_terminal_types": float(gbar_term @ gbar / g2),
       "SNR": float(g2 / N_full), "SNR_cf": float(g2 / N_rest), "SNR_cf_over_SNR": float(N_full / N_rest),
       "adam_P": P_adam, "adam_P_cf": P_cf, "adam_cf_over_P": P_cf / P_adam,
       "cos_term_vs_rest_mean": cos(gbar_term, gbar_rest),
       "norm_gbar": float(np.sqrt(g2)), "norm_gbar_terminal_types": float(np.linalg.norm(gbar_term)),
       "clip_frac": float(nrm[0] / B), "mean_grad_norm": float(nrm[1] / B), "B": B, "m": M, "K": K,
       "sec": time.time() - t0}
os.makedirs(f"{run}/gradattr", exist_ok=True)
with open(f"{run}/gradattr/{game}_{algo}_s{seed}_t{t}.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("game", "algo", "seed", "t", "pair_frac", "signal_share", "noise_share",
                                      "SNR", "SNR_cf_over_SNR", "adam_cf_over_P", "cos_term_vs_rest_mean", "clip_frac", "sec")}))
sys.stdout.flush(); os._exit(0)
