"""DIAG 2026-11-01: action-gap / greedy-agreement analysis on parameter checkpoints (DQN vs CW-Pair).
usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python gap_analysis.py <run_dir> <game> <seed> <t> [--mc]
For one (game, seed, checkpoint step t) and algos {dqn, cwpair_mlp, cwpair_lr}:
 1. state pool: each agent plays eps=0.05 from reset (64 envs x 800 steps, auto-reset); N states sampled per agent
    after step 100 -> pooled (3N states, same pool for every agent).
 2. scores on the pool: DQN Q(s,.), CW-Pair Borda(s,.).  Borda(a)-Borda(b) = A/(A-1) (Q(a)-Q(b)) if D is an exact
    difference, so Borda gaps are rescaled by (A-1)/A to Q units.  gap = top1 - top2 (Q units).
    Also the gap relative to the spread of state values across the pool (scale-free), where the state value for
    CW-Pair is v(s) = mean_r D((s,a*_s),(r,a*_r)) over 64 reference states (an estimate of V(s) - mean V(ref)).
 3. pairwise greedy agreement between the 3 agents on the pool.
 4. --mc: Monte-Carlo Q^pi(s,a) of each agent's own greedy policy (sticky env, R rollouts, horizon H, gamma .99)
    on a subset of the pool -> regret of the greedy action, top-1 accuracy, Spearman, predicted/MC gap ratio.
Writes <run_dir>/gap/<game>_s<seed>_t<t>.json."""
import json, os, pickle, sys, time
import numpy as np, jax, jax.numpy as jnp
import pgx
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rdqn import MinAtarTorso
from stoix.systems.q_learning.ff_rpair import PairNet
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import DiscreteQNetworkHead

run, game, seed, t = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
MC = "--mc" in sys.argv
N_PER, N_MC, R, H, G = 256, 128, 32, 400, 0.99
env = pgx.make(f"minatar-{game}"); env.sticky_action_prob = 0.1
A = env.num_actions
ALGOS = ["dqn", "cwpair_mlp", "cwpair_lr"]

def load(a):
    with open(f"{run}/ckpt/{game}_{a}_s{seed}_t{t}.pkl", "rb") as f:
        return pickle.load(f)

def scorer(a, p):
    if a == "dqn":
        net = FeedForwardActor(torso=MinAtarTorso(), action_head=DiscreteQNetworkHead(action_dim=A, epsilon=0.0))
        return jax.jit(lambda o: net.apply(p["q"], o).preferences), None
    net = PairNet(num_actions=A, head="free" if a == "cwpair_mlp" else "lowrank")
    sc = jax.jit(lambda o: net.apply(p, o, method=PairNet.scores) * (A - 1) / A)    # rescaled to Q units
    def vrel(o, oref):
        E = net.apply(p, o, method=PairNet.all_e); Er = net.apply(p, oref, method=PairNet.all_e)
        e = E[jnp.arange(E.shape[0]), jnp.argmax(sc(o), -1)]; er = Er[jnp.arange(Er.shape[0]), jnp.argmax(sc(oref), -1)]
        ex = jnp.broadcast_to(e[:, None], (e.shape[0], er.shape[0], e.shape[-1]))
        ey = jnp.broadcast_to(er[None], (e.shape[0], er.shape[0], e.shape[-1]))
        return net.apply(p, ex, ey, method=PairNet.D).mean(1)
    return sc, jax.jit(vrel)

P = {a: load(a) for a in ALGOS}
S = {a: scorer(a, P[a]) for a in ALGOS}
obs_of = lambda st: st.observation.astype(jnp.float32)

def reset_where(done, st, key):
    fresh = jax.vmap(env.init)(jax.random.split(key, done.shape[0]))
    return jax.tree_util.tree_map(lambda f, s: jnp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), f, s), fresh, st)

def collect(a, key, n_env=64, T=800, eps=0.05):
    sc = S[a][0]
    k0, key = jax.random.split(key)
    st = jax.vmap(env.init)(jax.random.split(k0, n_env))
    def body(carry, k):
        st = carry
        k1, k2, k3, k4 = jax.random.split(k, 4)
        g = jnp.argmax(sc(obs_of(st)), -1)
        rnd = jax.random.randint(k1, g.shape, 0, A)
        act = jnp.where(jax.random.uniform(k2, g.shape) < eps, rnd, g)
        nst = jax.vmap(env.step)(st, act, jax.random.split(k3, n_env))
        nst = reset_where(nst.terminated, nst, k4)
        return nst, st
    _, traj = jax.lax.scan(body, st, jax.random.split(key, T))
    flat = jax.tree_util.tree_map(lambda x: x[100:].reshape(((x.shape[0] - 100) * x.shape[1],) + x.shape[2:]), traj)
    idx = np.random.default_rng(seed * 7 + 1).choice((T - 100) * n_env, N_PER, replace=False)
    return jax.tree_util.tree_map(lambda x: x[idx], flat)

key = jax.random.PRNGKey(1000 + seed)
pools = []
for i, a in enumerate(ALGOS):
    pools.append(collect(a, jax.random.fold_in(key, i)))
pool = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, 0), *pools)
src = np.repeat(np.arange(len(ALGOS)), N_PER)                     # which agent's policy generated each state
O = obs_of(pool)
ref = O[np.random.default_rng(5).choice(O.shape[0], 64, replace=False)]

out = {"game": game, "seed": seed, "t": t, "A": A, "n_pool": int(O.shape[0]), "per_agent": {}, "agree": {}}
SC, GR = {}, {}
for a in ALGOS:
    sc, vrel = S[a]
    q = np.asarray(sc(O)); SC[a] = q; GR[a] = q.argmax(-1)
    srt = np.sort(q, -1); gap = srt[:, -1] - srt[:, -2]; spread_a = srt[:, -1] - srt[:, 0]
    v = np.asarray(q.max(-1)) if a == "dqn" else np.asarray(vrel(O, ref))
    if a == "dqn":
        v = v - np.asarray(sc(ref)).max(-1).mean()
    vstd = float(np.std(v))
    d = {"gap_mean": float(gap.mean()), "gap_median": float(np.median(gap)),
         "gap_q10": float(np.quantile(gap, .1)), "gap_q90": float(np.quantile(gap, .9)),
         "frac_gap_lt_0.01": float((gap < .01).mean()), "frac_gap_lt_0.1": float((gap < .1).mean()),
         "action_range_mean": float(spread_a.mean()), "state_value_std": vstd,
         "gap_over_vstd": float(gap.mean() / max(vstd, 1e-8)), "range_over_vstd": float(spread_a.mean() / max(vstd, 1e-8)),
         "level_mean": float(q.max(-1).mean()) if a == "dqn" else None,
         "greedy_action_hist": np.bincount(GR[a], minlength=A).tolist(),
         "gap_by_src": {ALGOS[j]: float(gap[src == j].mean()) for j in range(len(ALGOS))}}
    out["per_agent"][a] = d
for i in range(len(ALGOS)):
    for j in range(i + 1, len(ALGOS)):
        a, b = ALGOS[i], ALGOS[j]
        out["agree"][f"{a}|{b}"] = {"all": float((GR[a] == GR[b]).mean()),
                                    **{f"src_{ALGOS[k]}": float((GR[a] == GR[b])[src == k].mean()) for k in range(len(ALGOS))}}

if MC:
    sel = np.random.default_rng(11).choice(O.shape[0], N_MC, replace=False)
    sub = jax.tree_util.tree_map(lambda x: x[sel], pool)
    for a in ALGOS:
        sc = S[a][0]
        n = N_MC * A * R
        st = jax.tree_util.tree_map(lambda x: jnp.repeat(x, A * R, axis=0), sub)
        a0 = jnp.tile(jnp.repeat(jnp.arange(A), R), N_MC)
        @jax.jit
        def roll(st, k):
            k0, k = jax.random.split(k)
            st = jax.vmap(env.step)(st, a0, jax.random.split(k0, n))
            ret = st.rewards[:, 0].astype(jnp.float32); alive = 1.0 - st.terminated.astype(jnp.float32)
            def body(c, kk):
                st, ret, alive, disc = c
                act = jnp.argmax(sc(obs_of(st)), -1)
                nst = jax.vmap(env.step)(st, act, jax.random.split(kk, n))
                ret = ret + alive * disc * nst.rewards[:, 0].astype(jnp.float32)
                alive = alive * (1.0 - nst.terminated.astype(jnp.float32))
                return (nst, ret, alive, disc * G), None
            (st, ret, alive, _), _ = jax.lax.scan(body, (st, ret, alive, G), jax.random.split(k, H - 1))
            return ret
        t0 = time.time()
        ret = np.asarray(roll(st, jax.random.PRNGKey(77 + seed))).reshape(N_MC, A, R)
        Qmc, se = ret.mean(-1), ret.std(-1, ddof=1) / np.sqrt(R)
        pred = SC[a][sel]
        g = pred.argmax(-1); best = Qmc.argmax(-1)
        regret = Qmc.max(-1) - Qmc[np.arange(N_MC), g]
        srt = np.sort(Qmc, -1); mcgap = srt[:, -1] - srt[:, -2]
        ps = np.sort(pred, -1); pgap = ps[:, -1] - ps[:, -2]
        from scipy.stats import spearmanr
        rho = [spearmanr(pred[i], Qmc[i])[0] for i in range(N_MC) if np.ptp(Qmc[i]) > 1e-6 and np.ptp(pred[i]) > 1e-9]
        # Q-difference error of predicted differences vs MC differences (all action pairs)
        dp = pred[:, :, None] - pred[:, None, :]; dm = Qmc[:, :, None] - Qmc[:, None, :]
        iu = np.triu_indices(A, 1)
        dp, dm = dp[:, iu[0], iu[1]].ravel(), dm[:, iu[0], iu[1]].ravel()
        slope = float(np.sum(dp * dm) / max(np.sum(dm * dm), 1e-12))
        out["per_agent"][a]["mc"] = {
            "n_states": N_MC, "R": R, "H": H, "sec": time.time() - t0,
            "regret_mean": float(regret.mean()), "regret_median": float(np.median(regret)),
            "top1_acc": float((g == best).mean()),
            "top1_acc_within_2se": float((regret <= 2 * np.sqrt(se.max(-1) ** 2 + se[np.arange(N_MC), g] ** 2)).mean()),
            "spearman_mean": float(np.nanmean(rho)) if rho else None,
            "mc_gap_mean": float(mcgap.mean()), "pred_gap_mean": float(pgap.mean()),
            "diff_slope_pred_on_mc": slope, "diff_corr": float(np.corrcoef(dp, dm)[0, 1]),
            "mc_se_mean": float(se.mean()), "mc_value_mean": float(Qmc.max(-1).mean()),
            "pred_level_minus_mc": (float((pred.max(-1) - Qmc.max(-1)).mean()) if a == "dqn" else None)}
os.makedirs(f"{run}/gap", exist_ok=True)
with open(f"{run}/gap/{game}_s{seed}_t{t}.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else "...") for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in out.items()})[:3000])
