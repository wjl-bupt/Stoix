"""Same-state action-ranking test for Gym MuJoCo (v5) critics saved by mujoco_offpolicy.py.  2026-11-08.

For a checkpoint {"actor", "q", "args"}:
  1. states: greedy (noise-free) episodes of the agent's own actor from fixed reset seeds; every `stride`-th state
     is a candidate, n_states sampled uniformly.  The full MuJoCo integration state (mj_getState, mjSTATE_INTEGRATION:
     qpos, qvel, act, warm-start, ctrl, time, ...) is stored so the state can be restored exactly.
  2. candidate actions per state, in the actor's [-1, 1] space (scaled to the env bounds):
       local  : pi(s) and K-1 Gaussian perturbations pi(s) + N(0, sigma^2), clipped           (what the actor gradient sees)
       global : K uniform actions in [-1, 1]
  3. true values of the deterministic greedy policy by deterministic roll-outs (no TimeLimit, horizon H, gamma .99,
     failure terminations stop the return):  Qhat(s, a) = r_0 + sum_{t>=1} gamma^t r_t.
  4. metrics per state (averaged): Kendall tau-b between Q(s, .) and Qhat(s, .); pairwise sign accuracy on pairs with
     |dQhat| > tol (unweighted and |dQhat|-weighted); greedy regret of argmax_a Q among the candidates, also normalised
     by the spread max-min Qhat.  All invariant to a common shift / positive scale of Q(s, .).
A replay check is reported: from a stored state, re-stepping the logged action reproduces the logged next observation.

usage: JAX_PLATFORMS=cpu python rank_mujoco.py --ckpt X.pkl --out res.json [--n_states 100 --K 8 --H 1000]
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/home/junweiluo/Stoix/scripts_mujoco")
import gymnasium as gym  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402

from mujoco_offpolicy import DetActor, QNet  # noqa: E402

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def get_state(u):
    st = np.empty(mujoco.mj_stateSize(u.model, SPEC)); mujoco.mj_getState(u.model, u.data, st, SPEC); return st


def set_state(u, st):
    mujoco.mj_setState(u.model, u.data, st, SPEC); mujoco.mj_forward(u.model, u.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_states", type=int, default=100); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--H", type=int, default=1000); ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--episodes", type=int, default=5); ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    ck = pickle.load(open(a.ckpt, "rb")); args = ck["args"]
    assert args["base"] in ("ddpg", "td3"), "deterministic actors only"
    gamma = 0.99
    env = gym.make(args["env"]); u0 = env.unwrapped
    ad = env.action_space.shape[0]
    lo, hi = env.action_space.low.astype(np.float32), env.action_space.high.astype(np.float32)
    scale, bias = (hi - lo) / 2.0, (hi + lo) / 2.0
    actor, qnet = DetActor(ad), QNet()
    pi = jax.jit(lambda o: actor.apply(ck["actor"], o))                       # [-1, 1]
    qf = jax.jit(lambda o, act: qnet.apply(ck["q"][0], o, act * scale + bias))
    t0 = time.time()

    # ---- 1. states on the agent's own greedy episodes
    cand, rep_err = [], []
    for ep in range(a.episodes):
        o, _ = env.reset(seed=a.seed + ep); done = False; t = 0
        while not done:
            st = get_state(u0)
            act = np.asarray(pi(jnp.asarray(o[None], jnp.float32)))[0]
            if t % a.stride == 0:
                cand.append((o.copy(), st, act))
            o2, r, te, tr, _ = env.step(act * scale + bias)
            if t % (a.stride * 10) == 0:                                      # replay check
                v = gym.make(args["env"]).unwrapped; v.reset(seed=0); set_state(v, st)
                o3, _, _, _, _ = v.step(act * scale + bias); rep_err.append(float(np.max(np.abs(o3 - o2)))); v.close()
            o = o2; t += 1; done = te or tr
    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(cand), size=min(a.n_states, len(cand)), replace=False)

    # ---- 2./3. candidate actions and deterministic roll-outs (2K envs in lockstep)
    K2 = 2 * a.K
    envs = [gym.make(args["env"]).unwrapped for _ in range(K2)]
    for v in envs:
        v.reset(seed=0)
    out = {"local": [], "global": []}
    allQ = []
    for i in pick:
        o, st, act = cand[i]
        loc = np.clip(act[None] + rng.normal(0, a.sigma, (a.K, ad)), -1, 1); loc[0] = act
        glo = rng.uniform(-1, 1, (a.K, ad))
        A0 = np.concatenate([loc, glo]).astype(np.float32)
        Qn = np.asarray(qf(jnp.asarray(np.repeat(o[None], K2, 0), jnp.float32), jnp.asarray(A0)))
        G = np.zeros(K2); live = np.ones(K2, bool); obs = np.zeros((K2, o.shape[0]), np.float32); disc = 1.0
        for k, v in enumerate(envs):
            set_state(v, st)
            ob, r, te, _, _ = v.step(A0[k] * scale + bias); G[k] = r; live[k] = not te; obs[k] = ob
        for h in range(1, a.H):
            if not live.any():
                break
            disc *= gamma
            acts = np.asarray(pi(jnp.asarray(obs)))
            for k in np.flatnonzero(live):
                ob, r, te, _, _ = envs[k].step(acts[k] * scale + bias)
                G[k] += disc * r; obs[k] = ob
                if te:
                    live[k] = False
        allQ.append((Qn, G))
        for name, sl in (("local", slice(0, a.K)), ("global", slice(a.K, K2))):
            out[name].append((Qn[sl], G[sl]))

    from scipy.stats import kendalltau
    res = {"ckpt": os.path.basename(a.ckpt), "env": args["env"], "head": args["head"], "N": len(pick), "K": a.K,
           "H": a.H, "sigma": a.sigma, "replay_err_max": float(max(rep_err)) if rep_err else None}
    for name, L in out.items():
        taus, acc_n, acc_d, wacc_n, wacc_d, reg, nreg = [], 0, 0, 0.0, 0.0, [], []
        for Qn, Gh in L:
            tol = 1e-3 * max(1.0, np.ptp(Gh))
            if np.ptp(Gh) > 1e-9:
                t = kendalltau(Qn, Gh).statistic
                if np.isfinite(t):
                    taus.append(t)
            for x in range(len(Gh)):
                for y in range(x + 1, len(Gh)):
                    d = Gh[x] - Gh[y]
                    if abs(d) > tol:
                        ok = np.sign(Qn[x] - Qn[y]) == np.sign(d)
                        acc_n += ok; acc_d += 1; wacc_n += ok * abs(d); wacc_d += abs(d)
            r = Gh.max() - Gh[int(np.argmax(Qn))]; reg.append(r); nreg.append(r / max(np.ptp(Gh), 1e-9))
        res[name] = {"tau": float(np.mean(taus)) if taus else None, "pair_acc": acc_n / max(acc_d, 1),
                     "wpair_acc": wacc_n / max(wacc_d, 1e-12), "regret": float(np.mean(reg)),
                     "nregret": float(np.mean(nreg)), "qhat_spread": float(np.mean([np.ptp(g) for _, g in L])),
                     "qnet_spread": float(np.mean([np.ptp(q) for q, _ in L]))}
    res["wall"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w")); print(json.dumps(res))
    np.savez_compressed(a.out.replace(".json", ".npz"), Q=np.array([q for q, _ in allQ]), G=np.array([g for _, g in allQ]))


if __name__ == "__main__":
    main()
