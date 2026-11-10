"""Breakout 'imminent-miss' same-state action-ranking test.  2026-11-09.

Question: CW-Q-rho ranks actions more accurately than DQN on MinAtar Breakout overall (ranking_g16), yet scores
lower and its episodes stay short (~340 vs ~600 steps).  Is its ranking specifically worse in the states where the
ball is coming down and the action decides whether the episode ends?

Two modes
  bank : collect a SHARED test set of states from several checkpoints (eps-greedy, sticky .1), save to a pickle.
  eval : for one checkpoint, score its action ranking on (a) states from its own greedy policy and (b) the shared
         bank.  Ground truth = Monte Carlo Q^pi of THIS checkpoint's greedy policy (first action a, then greedy),
         K rollouts, horizon H, plus the first-termination step of every rollout (for the miss probability).

State classes (from the pgx internal state; paddle is on row 9, bricks only on rows 1-3 so a descending ball below
row 3 reaches row 9 in tau = 9 - ball_y steps, only reflecting off the side walls):
  desc_k : ball descending (dir 2/3), ball_y >= 4, tau = 9 - ball_y = k  (k = 1..5)
  other  : everything else (ascending ball or ball in the brick zone)
'critical' (decided by MC, per checkpoint): desc state where actions differ in miss probability:
  P_miss(a) = P(episode ends within tau + 2 steps | first action a, then greedy pi);  max_a - min_a >= 0.5
Metrics (pooled over states of a class): sig_acc (MC-significant pairs ranked right), surv_acc (pairs whose miss
probabilities differ by >= .5: is the safer action ranked higher), safe_rate (greedy action has P_miss within .1 of
the best action), top1, regret; all invariant to a common shift / positive scaling of Q(s, .).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/home/junweiluo/Stoix")
sys.path.insert(0, "/home/junweiluo/Stoix/scripts_rdqn/ranking")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from rank_minatar import build_q  # noqa: E402

import pgx  # noqa: E402

ENV = pgx.make("minatar-breakout")
ENV.sticky_action_prob = 0.1
A = int(ENV.num_actions)
STEP = jax.vmap(ENV.step)


def load_q(ckpt, jsonl):
    cfg = json.loads(open(jsonl).readline())["config"]
    kind = "pair" if "_qcw_" in os.path.basename(ckpt) else "dqn"     # same convention as run_ranking.sh
    params = pickle.load(open(ckpt, "rb"))
    params = jax.tree_util.tree_map(lambda x: x[0], params)
    qf = build_q(cfg, A, kind)
    return jax.jit(lambda o: qf(params, o.astype(jnp.float32))), kind, float(cfg["system"]["gamma"])


def rollout_states(q, key, B, T, eps):
    """eps-greedy trajectories from fresh starts; returns stacked states [T, B] and alive mask."""
    k0, key = jax.random.split(key)
    s = jax.vmap(ENV.init)(jax.random.split(k0, B))

    @jax.jit
    def run(s, key):
        def f(c, k):
            s, alive = c
            k1, k2, k3 = jax.random.split(k, 3)
            g = jnp.argmax(q(s.observation), -1)
            r = jax.random.randint(k1, (B,), 0, A)
            act = jnp.where(jax.random.uniform(k2, (B,)) < eps, r, g)
            s2 = STEP(s, act, jax.random.split(k3, B))
            return (s2, alive & ~(s2.terminated | s2.truncated)), (s, alive)
        _, (S, alive) = jax.lax.scan(f, (s, jnp.ones(B, bool)), jax.random.split(key, T))
        return S, alive
    return run(s, key)


def classify(S):
    """-> tau (1..5 for desc states, 0 otherwise)."""
    y = np.asarray(S._ball_y); d = np.asarray(S._ball_dir)
    desc = ((d == 2) | (d == 3)) & (y >= 4) & (y <= 8)
    return np.where(desc, 9 - y, 0)


def pick_states(S, alive, n_desc, n_other, rng, stride=1):
    tau = classify(S)
    al = np.asarray(alive).copy()
    if stride > 1:
        al[np.arange(al.shape[0]) % stride != 0] = False
    idx_d = np.argwhere(al & (tau > 0)); idx_o = np.argwhere(al & (tau == 0))
    pd = idx_d[rng.choice(len(idx_d), size=min(n_desc, len(idx_d)), replace=False)]
    po = idx_o[rng.choice(len(idx_o), size=min(n_other, len(idx_o)), replace=False)]
    pick = np.concatenate([pd, po])
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[pick[:, 0], pick[:, 1]], S))


def mc_eval(q, gamma, St, K, H, chunk, key):
    """Q^pi MC: first action a, then greedy.  Returns G [N, A, K] and first-termination step [N, A, K] (H = none)."""
    N = St._ball_y.shape[0]
    M = N * A * K
    S0 = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.asarray(x), A * K, axis=0), St)
    a0 = jnp.tile(jnp.repeat(jnp.arange(A), K), N)

    @jax.jit
    def mc(S0, a0, key):
        n = a0.shape[0]
        k0, key = jax.random.split(key)
        s = STEP(S0, a0, jax.random.split(k0, n))
        G = s.rewards[:, 0].astype(jnp.float32)
        live = ~(s.terminated | s.truncated)
        tend = jnp.where(live, H, 1)

        def f(c, x):
            s, G, live, disc, tend = c
            kk, t = x
            act = jnp.argmax(q(s.observation), -1)
            s2 = STEP(s, act, jax.random.split(kk, n))
            G = G + disc * live * s2.rewards[:, 0].astype(jnp.float32)
            ended = live & (s2.terminated | s2.truncated)
            tend = jnp.where(ended, t, tend)
            return (s2, G, live & ~ended, disc * gamma, tend), None
        (s, G, live, _, tend), _ = jax.lax.scan(
            f, (s, G, live, jnp.float32(gamma), tend), (jax.random.split(key, H - 1), jnp.arange(2, H + 1)))
        return G, tend
    Gs, Ts = [], []
    for i in range(0, M, chunk):
        key, kk = jax.random.split(key)
        sl = lambda x: x[i:i + chunk]  # noqa: E731
        g, t = mc(jax.tree_util.tree_map(sl, S0), a0[i:i + chunk], kk)
        Gs.append(np.asarray(g)); Ts.append(np.asarray(t))
    return np.concatenate(Gs).reshape(N, A, K), np.concatenate(Ts).reshape(N, A, K)


def metrics(Qnet, G, Tend, tau, mask):
    Qhat = G.mean(-1); se = G.std(-1, ddof=1) / np.sqrt(G.shape[-1])
    out = {"n": int(mask.sum())}
    if out["n"] == 0:
        return out
    pm = (Tend <= (tau[:, None, None] + 2)).mean(-1)                   # P_miss(s, a)  (only meaningful for desc)
    n_sig = n_ok = n_sv = n_svok = 0; reg, top, safe = [], [], []
    for i in np.where(mask)[0]:
        g = int(np.argmax(Qnet[i])); b = int(np.argmax(Qhat[i]))
        r = Qhat[i, b] - Qhat[i, g]; reg.append(r)
        top.append(r <= 2 * np.sqrt(se[i, b] ** 2 + se[i, g] ** 2) + 1e-12)
        safe.append(pm[i, g] <= pm[i].min() + 0.1)
        for x in range(A):
            for y in range(x + 1, A):
                d = Qhat[i, x] - Qhat[i, y]
                sq = np.sign(Qnet[i, x] - Qnet[i, y])
                if abs(d) > 2 * np.sqrt(se[i, x] ** 2 + se[i, y] ** 2) and abs(d) > 1e-9:
                    n_sig += 1; n_ok += int(sq == np.sign(d))
                dp = pm[i, y] - pm[i, x]                                # > 0: x is safer
                if abs(dp) >= 0.5:
                    n_sv += 1; n_svok += int(sq == np.sign(dp))
    out.update(sig_acc=n_ok / n_sig if n_sig else None, n_sig=n_sig, surv_acc=n_svok / n_sv if n_sv else None,
               n_surv=n_sv, safe_rate=float(np.mean(safe)), top1=float(np.mean(top)), regret=float(np.mean(reg)),
               pmiss_best=float(pm[mask].min(1).mean()), pmiss_worst=float(pm[mask].max(1).mean()))
    return out


def report(Qnet, G, Tend, St):
    tau = classify(St)
    pm = (Tend <= (tau[:, None, None] + 2)).mean(-1)
    crit = (tau > 0) & (pm.max(1) - pm.min(1) >= 0.5)
    res = {"all": metrics(Qnet, G, Tend, tau, np.ones_like(tau, bool)),
           "other": metrics(Qnet, G, Tend, tau, tau == 0),
           "desc": metrics(Qnet, G, Tend, tau, tau > 0),
           "critical": metrics(Qnet, G, Tend, tau, crit),
           "desc_noncrit": metrics(Qnet, G, Tend, tau, (tau > 0) & ~crit)}
    for k in range(1, 6):
        res[f"crit_tau{k}"] = metrics(Qnet, G, Tend, tau, crit & (tau == k))
    return res, crit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["bank", "eval"])
    ap.add_argument("--ckpts", nargs="+", default=[]); ap.add_argument("--jsonls", nargs="+", default=[])
    ap.add_argument("--bank", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--n_desc", type=int, default=320); ap.add_argument("--n_other", type=int, default=128)
    ap.add_argument("--K", type=int, default=32); ap.add_argument("--H", type=int, default=500)
    ap.add_argument("--B", type=int, default=64); ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.1); ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(a.seed); key = jax.random.PRNGKey(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.mode == "bank":                                               # shared state set from several policies
        parts = []
        for c, j in zip(a.ckpts, a.jsonls):
            q, _, _ = load_q(c, j)
            key, k = jax.random.split(key)
            S, alive = rollout_states(q, k, a.B, a.T, a.eps)
            parts.append(pick_states(S, alive, a.n_desc // len(a.ckpts), a.n_other // len(a.ckpts), rng, stride=3))
        bank = jax.tree_util.tree_map(lambda *x: np.concatenate(x), *parts)
        pickle.dump(bank, open(a.out, "wb"))
        print("bank", bank._ball_y.shape[0], "desc", int((classify(bank) > 0).sum()), time.time() - t0)
        return

    q, kind, gamma = load_q(a.ckpts[0], a.jsonls[0])
    res = {"ckpt": os.path.basename(a.ckpts[0]), "kind": kind, "K": a.K, "H": a.H}
    key, k = jax.random.split(key)
    S, alive = rollout_states(q, k, a.B, a.T, 0.0)                     # own greedy states
    own = pick_states(S, alive, a.n_desc, a.n_other, rng, stride=1)
    sets = {"own": own}
    if a.bank:
        sets["bank"] = pickle.load(open(a.bank, "rb"))
    save = {}
    for nm, St in sets.items():
        Qnet = np.asarray(q(jnp.asarray(St.observation)))
        key, k = jax.random.split(key)
        G, Tend = mc_eval(q, gamma, St, a.K, a.H, a.chunk, k)
        r, crit = report(Qnet, G, Tend, St)
        res[nm] = r
        save.update({f"{nm}_Qnet": Qnet, f"{nm}_G": G.astype(np.float32), f"{nm}_Tend": Tend.astype(np.int16),
                     f"{nm}_tau": classify(St), f"{nm}_pos": np.asarray(St._pos), f"{nm}_bx": np.asarray(St._ball_x),
                     f"{nm}_by": np.asarray(St._ball_y), f"{nm}_dir": np.asarray(St._ball_dir)})
    res["wall"] = time.time() - t0
    json.dump(res, open(a.out, "w"))
    np.savez_compressed(a.out.replace(".json", ".npz"), **save)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("all", "desc", "critical")})
                      for k, v in res.items()}, default=str)[:2000])


if __name__ == "__main__":
    main()
