"""DIAG 2026-11-01 (v2): action gaps and greedy agreement on checkpoints, DQN vs CW-Pair, all seeds at once.
usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python gap_analysis2.py <run_dir> <game> <t> [--mc]
Agents = {dqn, cwpair_mlp, cwpair_lr} x seeds that have a checkpoint at step t (normally 0-2).
 1. State pool: every agent plays eps=0.05 from reset (64 envs x 800 steps, auto-reset), 128 states sampled after
    step 100 -> pooled (9 x 128 states); every agent is scored on the same pool.
 2. Scores in Q units: DQN Q(s,.); CW-Pair Borda(s,.) x (A-1)/A (exact Q differences if D is an exact difference).
    gap = top1 - top2; range = max - min over actions; state-value spread v_std (DQN: max_a Q; CW-Pair:
    v(s) = mean_r D((s,a*_s),(r,a*_r)) over 64 reference states), gap/v_std is scale-free.
 3. Greedy agreement matrix over all 9 agents -> within-algorithm (cross-seed) vs cross-algorithm agreement.
 4. --mc: Monte-Carlo Q^pi(s,a) of each agent's own greedy policy on 96 pool states, 5 actions x R=32 rollouts,
    horizon 400, gamma .99, common random numbers (rollout r of every action of a state uses the same key stream),
    so action differences have paired SEs.  Reports regret of the greedy action, sign accuracy of the predicted
    action differences on pairs whose MC difference is significant (|diff| > 2 SE), top-1 accuracy on states whose
    MC best action is significant, MC vs predicted gap scale (calibration of action differences).
Params are jit arguments (compile once per algorithm).  Writes <run_dir>/gap2/<game>_t<t>.json."""
import glob, json, os, pickle, sys, time
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_compilation_cache_dir", "/home/junweiluo/.cache/jax_diag_gap")   # retries after XLA segfaults reuse compiled programs
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
import pgx
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rdqn import MinAtarTorso
from stoix.systems.q_learning.ff_rpair import PairNet
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import DiscreteQNetworkHead

run, game, t = sys.argv[1], sys.argv[2], int(sys.argv[3])
MC = "--mc" in sys.argv
N_PER, N_REF, N_MC, R, H, G = 128, 64, 96, 32, 400, 0.99
env = pgx.make(f"minatar-{game}"); env.sticky_action_prob = 0.1
A = env.num_actions
ALGOS = ["dqn", "cwpair_mlp", "cwpair_lr"]
SEEDS = sorted(s for s in range(10) if all(os.path.exists(f"{run}/ckpt/{game}_{a}_s{s}_t{t}.pkl") for a in ALGOS))
AG = [(a, s) for a in ALGOS for s in SEEDS]
P = {}
for a, s in AG:
    with open(f"{run}/ckpt/{game}_{a}_s{s}_t{t}.pkl", "rb") as f:
        P[(a, s)] = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))

dqn_net = FeedForwardActor(torso=MinAtarTorso(), action_head=DiscreteQNetworkHead(action_dim=A, epsilon=0.0))
pair_net = {"cwpair_mlp": PairNet(num_actions=A, head="free"), "cwpair_lr": PairNet(num_actions=A, head="lowrank")}

def score_fn(a):
    if a == "dqn":
        return lambda p, o: dqn_net.apply(p["q"], o).preferences
    net = pair_net[a]
    return lambda p, o: net.apply(p, o, method=PairNet.scores) * (A - 1) / A

SCORE = {a: jax.jit(score_fn(a)) for a in ALGOS}

def vrel_fn(a):
    sc = score_fn(a)
    if a == "dqn":
        return lambda p, o, oref: sc(p, o).max(-1) - sc(p, oref).max(-1).mean()
    net = pair_net[a]
    def f(p, o, oref):
        E = net.apply(p, o, method=PairNet.all_e); Er = net.apply(p, oref, method=PairNet.all_e)
        e = E[jnp.arange(E.shape[0]), jnp.argmax(sc(p, o), -1)]
        er = Er[jnp.arange(Er.shape[0]), jnp.argmax(sc(p, oref), -1)]
        ex = jnp.broadcast_to(e[:, None], (e.shape[0], er.shape[0], e.shape[-1]))
        ey = jnp.broadcast_to(er[None], (e.shape[0], er.shape[0], e.shape[-1]))
        return net.apply(p, ex, ey, method=PairNet.D).mean(1)
    return f

VREL = {a: jax.jit(vrel_fn(a)) for a in ALGOS}
obs_of = lambda st: st.observation.astype(jnp.float32)

def make_collect(a, n_env=64, T=800, eps=0.05):
    sc = score_fn(a)
    def collect(p, key):
        k0, key = jax.random.split(key)
        st = jax.vmap(env.init)(jax.random.split(k0, n_env))
        def body(st, k):
            k1, k2, k3, k4 = jax.random.split(k, 4)
            g = jnp.argmax(sc(p, obs_of(st)), -1)
            act = jnp.where(jax.random.uniform(k2, g.shape) < eps, jax.random.randint(k1, g.shape, 0, A), g)
            nst = jax.vmap(env.step)(st, act, jax.random.split(k3, n_env))
            fresh = jax.vmap(env.init)(jax.random.split(k4, n_env))
            nst = jax.tree_util.tree_map(
                lambda f_, s_: jnp.where(nst.terminated.reshape((-1,) + (1,) * (s_.ndim - 1)), f_, s_), fresh, nst)
            return nst, st
        _, traj = jax.lax.scan(body, st, jax.random.split(key, T))
        return jax.tree_util.tree_map(lambda x: x[100:].reshape(((T - 100) * n_env,) + x.shape[2:]), traj)
    return jax.jit(collect), (T - 100) * n_env

def make_mc(a):
    sc = score_fn(a)
    def roll(p, st, a0, base):                       # base: per-env key (CRN: identical across the A actions)
        n = a0.shape[0]
        k0 = jax.vmap(lambda k: jax.random.fold_in(k, 0))(base)
        st = jax.vmap(env.step)(st, a0, k0)
        ret = st.rewards[:, 0].astype(jnp.float32); alive = 1.0 - st.terminated.astype(jnp.float32)
        def body(c, i):
            st, ret, alive, disc = c
            act = jnp.argmax(sc(p, obs_of(st)), -1)
            ks = jax.vmap(lambda k: jax.random.fold_in(k, i))(base)
            nst = jax.vmap(env.step)(st, act, ks)
            ret = ret + alive * disc * nst.rewards[:, 0].astype(jnp.float32)
            alive = alive * (1.0 - nst.terminated.astype(jnp.float32))
            return (nst, ret, alive, disc * G), None
        (st, ret, alive, _), _ = jax.lax.scan(body, (st, ret, alive, jnp.float32(G)), jnp.arange(1, H))
        return ret
    return jax.jit(roll)

t0 = time.time()
pools, src = [], []
rng = np.random.default_rng(t % 100003)
COLLECT = {a: make_collect(a) for a in ALGOS}                       # one compiled program per algorithm
for i, (a, s) in enumerate(AG):
    col, n_all = COLLECT[a]
    traj = col(P[(a, s)], jax.random.PRNGKey(1000 * s + 17 * i + 3))
    idx = rng.choice(n_all, N_PER, replace=False)
    pools.append(jax.tree_util.tree_map(lambda x: x[idx], traj)); src += [i] * N_PER
pool = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, 0), *pools); src = np.array(src)
O = obs_of(pool); ref = O[rng.choice(O.shape[0], N_REF, replace=False)]
t_pool = time.time() - t0

out = {"game": game, "t": t, "A": A, "seeds": SEEDS, "agents": [f"{a}_s{s}" for a, s in AG], "n_pool": int(O.shape[0]),
       "per_agent": {}, "agree_matrix": None, "agree_summary": {}, "timing": {"pool_sec": t_pool}}
SC, GR = {}, {}
for i, (a, s) in enumerate(AG):
    q = np.asarray(SCORE[a](P[(a, s)], O)); SC[i] = q; GR[i] = q.argmax(-1)
    srt = np.sort(q, -1); gap = srt[:, -1] - srt[:, -2]; rng_a = srt[:, -1] - srt[:, 0]
    v = np.asarray(VREL[a](P[(a, s)], O, ref)); vstd = float(v.std())
    own = src == i
    out["per_agent"][f"{a}_s{s}"] = {
        "algo": a, "seed": s, "gap_mean": float(gap.mean()), "gap_median": float(np.median(gap)),
        "range_mean": float(rng_a.mean()), "v_std": vstd, "gap_over_vstd": float(gap.mean() / max(vstd, 1e-8)),
        "range_over_vstd": float(rng_a.mean() / max(vstd, 1e-8)), "gap_own_states": float(gap[own].mean()),
        "frac_gap_lt_1pct_vstd": float((gap < 0.01 * vstd).mean()),
        "level_maxq_mean": float(q.max(-1).mean()) if a == "dqn" else None,
        "greedy_hist": np.bincount(GR[i], minlength=A).tolist()}
n = len(AG); M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        M[i, j] = (GR[i] == GR[j]).mean()
out["agree_matrix"] = M.tolist()
for x in ALGOS:
    for y in ALGOS:
        I = [i for i, (a, _) in enumerate(AG) if a == x]; J = [j for j, (a, _) in enumerate(AG) if a == y]
        vals = [M[i, j] for i in I for j in J if i != j]
        if x == y:
            out["agree_summary"][f"{x}|{x} (cross-seed)"] = float(np.mean(vals)) if vals else None
        elif ALGOS.index(x) < ALGOS.index(y):
            same = [M[i, j] for i in I for j in J if AG[i][1] == AG[j][1]]
            out["agree_summary"][f"{x}|{y}"] = float(np.mean(vals))
            out["agree_summary"][f"{x}|{y} (same seed)"] = float(np.mean(same))
out["agree_summary"]["chance_uniform"] = 1.0 / A

if MC:
    t1 = time.time()
    sel = rng.choice(O.shape[0], N_MC, replace=False)
    sub = jax.tree_util.tree_map(lambda x: x[sel], pool)
    nn_ = N_MC * A * R
    st = jax.tree_util.tree_map(lambda x: jnp.repeat(x, A * R, axis=0), sub)          # order (state, action, r)
    a0 = jnp.tile(jnp.repeat(jnp.arange(A), R), N_MC)
    K = jax.random.split(jax.random.PRNGKey(t % 9973 + 7), N_MC * R).reshape(N_MC, 1, R, 2)
    base = jnp.broadcast_to(K, (N_MC, A, R, 2)).reshape(nn_, 2)                     # CRN across actions
    ROLL = {a: make_mc(a) for a in ALGOS}
    iu = np.triu_indices(A, 1)
    for i, (a, s) in enumerate(AG):
        ret = np.asarray(ROLL[a](P[(a, s)], st, a0, base)).reshape(N_MC, A, R)
        Qmc = ret.mean(-1)
        dif = ret[:, :, None, :] - ret[:, None, :, :]                                   # [N, A, A, R] paired
        dm, dse = dif.mean(-1), dif.std(-1, ddof=1) / np.sqrt(R)
        pred = SC[i][sel]; dp = pred[:, :, None] - pred[:, None, :]
        g = pred.argmax(-1); best = Qmc.argmax(-1)
        regret = Qmc.max(-1) - Qmc[np.arange(N_MC), g]
        sig = np.abs(dm) > 2 * dse                                                      # significant MC differences
        sig_u = sig[:, iu[0], iu[1]]; dm_u = dm[:, iu[0], iu[1]]; dp_u = dp[:, iu[0], iu[1]]
        sign_acc = float((np.sign(dp_u[sig_u]) == np.sign(dm_u[sig_u])).mean()) if sig_u.any() else None
        srtq = np.sort(Qmc, -1); second = np.argsort(Qmc, -1)[:, -2]
        best_sig = dm[np.arange(N_MC), best, second] > 2 * dse[np.arange(N_MC), best, second]
        dm_flat, dp_flat = dm_u.ravel(), dp_u.ravel()
        out["per_agent"][f"{a}_s{s}"]["mc"] = {
            "n_states": N_MC, "R": R, "H": H,
            "regret_mean": float(regret.mean()), "regret_over_mcgap": float(regret.mean() / max((srtq[:, -1] - srtq[:, -2]).mean(), 1e-8)),
            "top1_acc": float((g == best).mean()),
            "frac_states_best_sig": float(best_sig.mean()),
            "top1_acc_on_sig_states": float((g == best)[best_sig].mean()) if best_sig.any() else None,
            "frac_pairs_sig": float(sig_u.mean()), "sign_acc_on_sig_pairs": sign_acc,
            "mc_absdiff_mean": float(np.abs(dm_flat).mean()), "pred_absdiff_mean": float(np.abs(dp_flat).mean()),
            "scale_ratio_pred_over_mc": float(np.abs(dp_flat).mean() / max(np.abs(dm_flat).mean(), 1e-8)),
            "diff_corr": float(np.corrcoef(dp_flat, dm_flat)[0, 1]),
            "diff_slope_pred_on_mc": float(np.sum(dp_flat * dm_flat) / max(np.sum(dm_flat ** 2), 1e-12)),
            "mc_gap_mean": float((srtq[:, -1] - srtq[:, -2]).mean()), "mc_pair_se_mean": float(dse[:, iu[0], iu[1]].mean()),
            "mc_value_mean": float(Qmc.max(-1).mean())}
    out["timing"]["mc_sec"] = time.time() - t1
out["timing"]["total_sec"] = time.time() - t0
os.makedirs(f"{run}/gap2", exist_ok=True)
with open(f"{run}/gap2/{game}_t{t}.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["agree_summary"]), json.dumps(out["timing"]))

sys.stdout.flush(); os._exit(0)   # JAX sometimes segfaults at interpreter shutdown; the json is already written
