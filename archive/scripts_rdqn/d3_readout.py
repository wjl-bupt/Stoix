"""D3 (2026-11-03): read-out swap on CW-Pair MLP K=8 checkpoints, no retraining.
usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python d3_readout.py <run_dir> <game> <seed> <t>
Read-outs (score of action a at state s; greedy = argmax):
  R0  Borda (as trained):  (1/(A-1)) sum_b D((s,a),(s,b)) * (A-1)/A    -- same-state pairs, never in training
  R1  anchor:              mean_{z in Z1} D((s,a), z)                  -- cross-state pairs, the training support
  R1s antisym. anchor:     mean_z 0.5 [D((s,a),z) - D(z,(s,a))]
  (R1 with a disjoint anchor set Z2 only to measure read-out stability.)
  Z1, Z2 = 64 replay-like transitions (state, taken action) each, from the agent's own pool, disjoint from evaluated states.
  Note: a least-squares (HodgeRank) potential on the graph {eval nodes} x Z  U  Z x Z ranks actions exactly like R1s
  (f_(s,a) = mean_z (d_(s,a),z + f_z)), so it is not a separate read-out.
1. Pool: agent's own Borda policy, eps=0.05, 64 envs x 800 steps (first 100 dropped).
2. MC (as gap_analysis2): 96 states x A actions x 32 rollouts (common random numbers across actions), horizon 400,
   gamma .99, rollouts follow the trained (Borda) greedy policy -> Q^pi(s,a); per read-out: regret, top-1 on states with a
   significant best action, sign accuracy on significant pairs, scale ratio |pred diff| / |MC diff|, correlation.
3. Control: each read-out as an eps=0 greedy policy, 128 envs, sticky 0.1, first episode per env (cap 6000 steps).
4. Consistency: on the 96 MC states, same-state antisym. D_ss(a,b)=0.5[D((s,a),(s,b))-D((s,b),(s,a))] vs the R1s
   difference; cross-edge asymmetry |D(x,z)+D(z,x)| / |D(x,z)|; argmax agreement R0/R1/R1s/R1(Z2) on 1024 pool states.
Writes <run_dir>/d3/<game>_s<seed>_t<t>.json"""
import json, os, pickle, sys, time
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_compilation_cache_dir", "/home/junweiluo/.cache/jax_d3")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
import pgx
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rpair import PairNet

run, game, seed, t = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
NZ, N_MC, R, H, G, N_AGR, NEV, CAP = 64, 96, 32, 400, 0.99, 1024, 128, 6000
env = pgx.make(f"minatar-{game}"); env.sticky_action_prob = 0.1
A = env.num_actions
with open(f"{run}/ckpt/{game}_cwpair_mlp_s{seed}_t{t}.pkl", "rb") as f:
    P = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
net = PairNet(num_actions=A, head="free")
obs_of = lambda st: st.observation.astype(jnp.float32)

def borda(p, o):
    return net.apply(p, o, method=PairNet.scores) * (A - 1) / A
def ez_of(p, o, a):
    return net.apply(p, net.apply(p, o, method=PairNet.phi), a, method=PairNet.sa)
def anchor(p, o, ez, sym):
    E = net.apply(p, o, method=PairNet.all_e)                                       # [N, A, E]
    N, Z = E.shape[0], ez.shape[0]
    ex = jnp.broadcast_to(E[:, :, None, :], (N, A, Z, E.shape[-1]))
    ey = jnp.broadcast_to(ez[None, None], (N, A, Z, ez.shape[-1]))
    d = net.apply(p, ex, ey, method=PairNet.D).mean(-1)
    if sym:
        d = 0.5 * (d - net.apply(p, ey, ex, method=PairNet.D).mean(-1))
    return d
READ = {"R0": lambda p, o, ez: borda(p, o), "R1": lambda p, o, ez: anchor(p, o, ez, False),
        "R1s": lambda p, o, ez: anchor(p, o, ez, True)}

# ---------------- 1. pool ----------------
def collect(p, key, n_env=64, T=800, eps=0.05):
    k0, key = jax.random.split(key)
    st = jax.vmap(env.init)(jax.random.split(k0, n_env))
    def body(st, k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        g = jnp.argmax(borda(p, obs_of(st)), -1)
        act = jnp.where(jax.random.uniform(k2, g.shape) < eps, jax.random.randint(k1, g.shape, 0, A), g)
        nst = jax.vmap(env.step)(st, act, jax.random.split(k3, n_env))
        fresh = jax.vmap(env.init)(jax.random.split(k4, n_env))
        nst2 = jax.tree_util.tree_map(
            lambda f_, s_: jnp.where(nst.terminated.reshape((-1,) + (1,) * (s_.ndim - 1)), f_, s_), fresh, nst)
        return nst2, (st, act)
    _, (traj, acts) = jax.lax.scan(body, st, jax.random.split(key, T))
    cut = lambda x: x[100:].reshape(((T - 100) * n_env,) + x.shape[2:])
    return jax.tree_util.tree_map(cut, traj), cut(acts)

t0 = time.time()
pool, pacts = jax.jit(collect)(P, jax.random.PRNGKey(1000 * seed + t % 9973 + 11))
Npool = pacts.shape[0]
rng = np.random.default_rng(seed * 7919 + t % 10007)
perm = rng.permutation(Npool)
i_z1, i_z2, i_mc, i_agr = perm[:NZ], perm[NZ:2 * NZ], perm[2 * NZ:2 * NZ + N_MC], perm[2 * NZ + N_MC:2 * NZ + N_MC + N_AGR]
O = obs_of(pool)
ez1 = jax.jit(ez_of)(P, O[i_z1], pacts[i_z1]); ez2 = jax.jit(ez_of)(P, O[i_z2], pacts[i_z2])
SC = {k: jax.jit(f) for k, f in READ.items()}
out = {"game": game, "seed": seed, "t": t, "A": A, "n_pool": int(Npool), "readouts": {}, "timing": {"pool": time.time() - t0}}

# ---------------- 4. agreement & consistency ----------------
def chunked(f, o, ez, n=128):
    return np.concatenate([np.asarray(f(P, o[i:i + n], ez)) for i in range(0, o.shape[0], n)], 0)
Oa = O[i_agr]
ga = {k: chunked(SC[k], Oa, ez1).argmax(-1) for k in SC}
ga["R1_Z2"] = chunked(SC["R1"], Oa, ez2).argmax(-1)
out["agreement"] = {f"{a}|{b}": float((ga[a] == ga[b]).mean()) for a, b in
                    [("R0", "R1"), ("R0", "R1s"), ("R1", "R1s"), ("R1", "R1_Z2")]}
out["agreement"]["chance"] = 1.0 / A
Om = O[i_mc]
E = net.apply(P, Om, method=PairNet.all_e)                                           # [N, A, E]
ea = jnp.broadcast_to(E[:, :, None], (N_MC, A, A, E.shape[-1])); eb = jnp.swapaxes(ea, 1, 2)
Dss = np.asarray(net.apply(P, ea, eb, method=PairNet.D))                             # D((s,a),(s,b))
Dss_anti = 0.5 * (Dss - np.swapaxes(Dss, 1, 2))
r1s = np.asarray(SC["R1s"](P, Om, ez1)); dfr = r1s[:, :, None] - r1s[:, None, :]
iu = np.triu_indices(A, 1)
x_, y_ = Dss_anti[:, iu[0], iu[1]].ravel(), dfr[:, iu[0], iu[1]].ravel()
ex_ = jnp.broadcast_to(E[:, :, None, :], (N_MC, A, NZ, E.shape[-1])); ey_ = jnp.broadcast_to(ez1[None, None], ex_.shape)
Dxz = np.asarray(net.apply(P, ex_, ey_, method=PairNet.D)); Dzx = np.asarray(net.apply(P, ey_, ex_, method=PairNet.D))
out["consistency"] = {
    "same_state_absdiff_mean": float(np.abs(x_).mean()), "anchor_absdiff_mean": float(np.abs(y_).mean()),
    "same_over_anchor_scale": float(np.abs(x_).mean() / max(np.abs(y_).mean(), 1e-12)),
    "same_vs_anchor_corr": float(np.corrcoef(x_, y_)[0, 1]),
    "same_vs_anchor_rel_sqerr": float(np.sum((x_ - y_) ** 2) / max(np.sum(y_ ** 2), 1e-12)),
    "cross_edge_asymmetry": float(np.abs(Dxz + Dzx).mean() / max(np.abs(Dxz).mean(), 1e-12)),
    "cross_edge_absD_mean": float(np.abs(Dxz).mean())}

# ---------------- 2. MC ----------------
t1 = time.time()
def roll(p, st, a0, base):
    k0 = jax.vmap(lambda k: jax.random.fold_in(k, 0))(base)
    st = jax.vmap(env.step)(st, a0, k0)
    ret = st.rewards[:, 0].astype(jnp.float32); alive = 1.0 - st.terminated.astype(jnp.float32)
    def body(c, i):
        st, ret, alive, disc = c
        act = jnp.argmax(borda(p, obs_of(st)), -1)
        ks = jax.vmap(lambda k: jax.random.fold_in(k, i))(base)
        nst = jax.vmap(env.step)(st, act, ks)
        ret = ret + alive * disc * nst.rewards[:, 0].astype(jnp.float32)
        alive = alive * (1.0 - nst.terminated.astype(jnp.float32))
        return (nst, ret, alive, disc * G), None
    (st, ret, alive, _), _ = jax.lax.scan(body, (st, ret, alive, jnp.float32(G)), jnp.arange(1, H))
    return ret
sub = jax.tree_util.tree_map(lambda x: x[i_mc], pool)
nn_ = N_MC * A * R
st = jax.tree_util.tree_map(lambda x: jnp.repeat(x, A * R, axis=0), sub)
a0 = jnp.tile(jnp.repeat(jnp.arange(A), R), N_MC)
K = jax.random.split(jax.random.PRNGKey(t % 9973 + 7 + 100 * seed), N_MC * R).reshape(N_MC, 1, R, 2)
base = jnp.broadcast_to(K, (N_MC, A, R, 2)).reshape(nn_, 2)
ret = np.asarray(jax.jit(roll)(P, st, a0, base)).reshape(N_MC, A, R)
Qmc = ret.mean(-1)
dif = ret[:, :, None, :] - ret[:, None, :, :]
dm, dse = dif.mean(-1), dif.std(-1, ddof=1) / np.sqrt(R)
best = Qmc.argmax(-1); second = np.argsort(Qmc, -1)[:, -2]
best_sig = dm[np.arange(N_MC), best, second] > 2 * dse[np.arange(N_MC), best, second]
sig_u = (np.abs(dm) > 2 * dse)[:, iu[0], iu[1]]; dm_u = dm[:, iu[0], iu[1]]
srtq = np.sort(Qmc, -1)
out["mc"] = {"n_states": N_MC, "frac_states_best_sig": float(best_sig.mean()), "frac_pairs_sig": float(sig_u.mean()),
             "mc_gap_mean": float((srtq[:, -1] - srtq[:, -2]).mean()), "mc_absdiff_mean": float(np.abs(dm_u).mean()),
             "mc_pair_se_mean": float(dse[:, iu[0], iu[1]].mean())}
preds = {k: np.asarray(SC[k](P, Om, ez1)) for k in SC}
preds["R1_Z2"] = np.asarray(SC["R1"](P, Om, ez2))
for k, pred in preds.items():
    dp = pred[:, :, None] - pred[:, None, :]; dp_u = dp[:, iu[0], iu[1]]
    g = pred.argmax(-1); regret = Qmc.max(-1) - Qmc[np.arange(N_MC), g]
    out["readouts"].setdefault(k, {})["mc"] = {
        "regret_mean": float(regret.mean()), "top1_acc": float((g == best).mean()),
        "top1_acc_on_sig_states": float((g == best)[best_sig].mean()) if best_sig.any() else None,
        "sign_acc_on_sig_pairs": float((np.sign(dp_u[sig_u]) == np.sign(dm_u[sig_u])).mean()) if sig_u.any() else None,
        "scale_ratio_pred_over_mc": float(np.abs(dp_u).mean() / max(np.abs(dm_u).mean(), 1e-8)),
        "diff_corr": float(np.corrcoef(dp_u.ravel(), dm_u.ravel())[0, 1]),
        "diff_slope_pred_on_mc": float(np.sum(dp_u * dm_u) / max(np.sum(dm_u ** 2), 1e-12))}
out["timing"]["mc"] = time.time() - t1

# ---------------- 3. control: greedy episodes per read-out ----------------
t2 = time.time()
def make_play(k):
    f = READ[k]
    def play(p, ez, key):
        k0, key = jax.random.split(key)
        st = jax.vmap(env.init)(jax.random.split(k0, NEV))
        def body(c, kk):
            st, ret, length, done = c
            act = jnp.argmax(f(p, obs_of(st), ez), -1)
            nst = jax.vmap(env.step)(st, act, jax.random.split(kk, NEV))
            live = 1.0 - done
            ret = ret + live * nst.rewards[:, 0].astype(jnp.float32); length = length + live
            done = jnp.maximum(done, nst.terminated.astype(jnp.float32))
            return (nst, ret, length, done), None
        z = jnp.zeros(NEV)
        (st, ret, length, done), _ = jax.lax.scan(body, (st, z, z, z), jax.random.split(key, CAP))
        return ret, length, done
    return jax.jit(play)
for k in READ:
    r_, l_, d_ = [np.asarray(x) for x in make_play(k)(P, ez1, jax.random.PRNGKey(4242 + seed))]   # same env keys for all read-outs
    out["readouts"][k]["eval"] = {"return_mean": float(r_.mean()), "return_se": float(r_.std(ddof=1) / np.sqrt(NEV)),
                                  "length_mean": float(l_.mean()), "frac_finished": float(d_.mean())}
out["timing"]["eval"] = time.time() - t2
# training-log eval at this checkpoint (sanity check of R0 eval) and DQN reference
for a in ("cwpair_mlp", "dqn"):
    f = f"{run}/{game}_{a}_s{seed}.jsonl"
    if os.path.exists(f):
        rows = [json.loads(l) for l in open(f) if '"timestep"' in l]
        r = [x for x in rows if x["timestep"] == t]
        out[f"train_log_eval_{a}"] = r[0].get("eval_return") if r else None
out["timing"]["total"] = time.time() - t0
os.makedirs(f"{run}/d3", exist_ok=True)
json.dump(out, open(f"{run}/d3/{game}_s{seed}_t{t}.json", "w"), indent=1)
print(json.dumps({"agreement": out["agreement"], "consistency": out["consistency"],
                  **{k: (v["mc"]["regret_mean"], v["mc"]["scale_ratio_pred_over_mc"], v.get("eval", {}).get("return_mean")) for k, v in out["readouts"].items()},
                  "train_log": out.get("train_log_eval_cwpair_mlp"), "timing": out["timing"]}))
sys.stdout.flush(); os._exit(0)
