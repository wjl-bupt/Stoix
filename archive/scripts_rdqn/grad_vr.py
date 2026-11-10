"""DIAG 2026-11-02: offline gradient quality of the variance-reduction estimators on CW-Pair MLP K=8 checkpoints.
usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python grad_vr.py <run_dir> <game> <seed> <t>
Same replay proxy as grad_attr.py (own eps=0.1 policy, 102,400 transitions, target := online).  For each estimator
  plain        : uniform batch m=512, K=8 cyclic partners, mean over pairs      (the training loss)
  strat        : n_T=32 terminal items drawn from the terminal stratum + TT cycle, cell weights
  multi        : uniform batch + 56 extra partners per terminal item (row and column), row/column weights
  strat_multi  : both
B=256 minibatch gradients -> gbar, SNR = |gbar|^2 / sum Var, Adam proxy P = sum gbar^2 / sqrt(E g^2),
and agreement of gbar with the plain gbar (cos, |diff|/|gbar_plain|; all estimators are unbiased for the same target,
so differences must be within noise: reported with the noise level sqrt(sum Var / B))."""
import json, os, pickle, sys, time
import numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
jax.config.update("jax_compilation_cache_dir", "/home/junweiluo/.cache/jax_gradattr")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
import pgx
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss, strat_sample, build_weighted_pairs, build_rowcol_pairs
from stoix.systems.q_learning.dqn_types import Transition
run, game, seed, t = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
G, EPS, NENV, BURN, T, M, K, B, MAXR = 0.99, 0.1, 128, 300, 800, 512, 8, 256, 1000.0
env = pgx.make(f"minatar-{game}"); env.sticky_action_prob = 0.1
A = env.num_actions
with open(f"{run}/ckpt/{game}_cwpair_mlp_s{seed}_t{t}.pkl", "rb") as f:
    p = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
obs_of = lambda st: st.observation.astype(jnp.float32)
net = PairNet(num_actions=A, head="free")
score = lambda pp, o: net.apply(pp, o, method=PairNet.scores)
loss_fn, compute = make_pair_loss(net, G, "cross", 0.33, MAXR, n_partners=K)
def collect(pp, key):
    k0, key = jax.random.split(key)
    st = jax.vmap(env.init)(jax.random.split(k0, NENV))
    def body(st, k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        o = obs_of(st); gq = jnp.argmax(score(pp, o), -1)
        act = jnp.where(jax.random.uniform(k2, gq.shape) < EPS, jax.random.randint(k1, gq.shape, 0, A), gq)
        nst = jax.vmap(env.step)(st, act, jax.random.split(k3, NENV))
        rec = Transition(obs=o, action=act, reward=nst.rewards[:, 0].astype(jnp.float32), done=nst.terminated,
                         next_obs=obs_of(nst), info={})
        fresh = jax.vmap(env.init)(jax.random.split(k4, NENV))
        nst = jax.tree_util.tree_map(
            lambda f_, s_: jnp.where(nst.terminated.reshape((-1,) + (1,) * (s_.ndim - 1)), f_, s_), fresh, nst)
        return nst, rec
    _, traj = jax.lax.scan(body, st, jax.random.split(key, BURN + T))
    return jax.tree_util.tree_map(lambda x: x[BURN:].reshape((T * NENV,) + x.shape[2:]), traj)
t0 = time.time()
pool = jax.jit(collect)(p, jax.random.PRNGKey(7919 * seed + t % 10007))    # same pool as grad_attr.py
Np = pool.done.shape[0]; valid = jnp.ones(Np, bool); piT = pool.done.mean()
flat0, _ = ravel_pytree(p); nparam = flat0.shape[0]
def grad_one(key, mode):
    k1, k2, k3 = jax.random.split(key, 3)
    sh = jax.random.randint(k3, (), 1, M)
    if mode in ("strat", "strat_multi"):
        tr, pi = strat_sample(pool, valid, k1, M, 32)
    else:
        idx = jax.random.randint(k1, (M,), 0, Np); tr = jax.tree_util.tree_map(lambda x: x[idx], pool); pi = piT
    if mode == "plain":
        return ravel_pytree(jax.grad(lambda q: loss_fn(q, p, tr, sh)[0])(p))[0]
    if mode == "multi":
        ix, iy, pw = build_rowcol_pairs(k2, tr.done, K, sh, 56, 32)
    else:
        ix, iy, pw = build_weighted_pairs(k2, tr.done, pi, K, sh, 56 if mode == "strat_multi" else 0, 32, tt=True)
    return ravel_pytree(jax.grad(lambda q: compute(q, p, tr, ix, iy, pw=pw)[0])(p))[0]
out = {"game": game, "seed": seed, "t": t, "pi_T_pool": float(piT), "arms": {}}
gbar_plain = None
for mode in ("plain", "strat", "multi", "strat_multi"):
    f = jax.jit(lambda acc, k: (lambda g: (acc[0] + g, acc[1] + g ** 2))(grad_one(k, mode)))
    acc = (jnp.zeros(nparam), jnp.zeros(nparam))
    for k in jax.random.split(jax.random.PRNGKey(101 + seed), B):
        acc = f(acc, k)
    S, S2 = [np.asarray(x, np.float64) for x in acc]
    gbar = S / B; var = np.maximum(S2 / B - gbar ** 2, 0) * B / (B - 1)
    N = var.sum(); g2 = gbar @ gbar
    P = float(np.sum(gbar ** 2 / np.sqrt(var + gbar ** 2 + 1e-30)))
    if mode == "plain": gbar_plain, P_plain, N_plain = gbar, P, N
    out["arms"][mode] = {"SNR": float(g2 / N), "adam_P": P, "adam_P_over_plain": P / P_plain,
                         "noise_over_plain": float(N / N_plain),
                         "cos_vs_plain": float(gbar @ gbar_plain / (np.linalg.norm(gbar) * np.linalg.norm(gbar_plain))),
                         "diff_over_plain_norm": float(np.linalg.norm(gbar - gbar_plain) / np.linalg.norm(gbar_plain)),
                         "noise_level_of_diff": float(np.sqrt((N + N_plain) / B) / np.linalg.norm(gbar_plain))}
out["sec"] = time.time() - t0
os.makedirs(f"{run}/gradvr", exist_ok=True)
json.dump(out, open(f"{run}/gradvr/{game}_s{seed}_t{t}.json", "w"), indent=1)
print(json.dumps(out)); sys.stdout.flush(); os._exit(0)
