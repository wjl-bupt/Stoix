"""Same-state action-ranking test for MinAtar (pgx) Q networks.  2026-11-08.

For a saved checkpoint (dqn via ff_rdqn, qcw via ff_rpair) this script
  1. collects states visited by the agent's own greedy policy (eps = 0, sticky .1, as in evaluation),
  2. estimates the true action values of that policy by Monte Carlo:
       Qhat(s, a) = mean_k [ r_0 + sum_{t>=1} gamma^t r_t ],  first action a, then greedy pi, K rollouts with fresh
       env keys (sticky-action / spawn randomness), horizon H (gamma^H residual ignored),
  3. compares the network's ordering of actions in each state with the MC ordering.
Metrics (per state, then averaged over states):
  tau        Kendall tau-b between Q(s, .) and Qhat(s, .)        (states where Qhat is constant are skipped)
  sig_acc    fraction of action pairs whose MC difference is significant (|dQhat| > 2 SE) and whose sign the
             network gets right                                     (pooled over all significant pairs)
  regret     Qhat(s, a*) - Qhat(s, argmax_a Q(s, a)) with a* = argmax Qhat (value lost by the greedy choice)
  top1       fraction of states with regret <= 2 SE (greedy choice statistically indistinguishable from the best)
All metrics are invariant to a common shift / positive scaling of Q(s, .), so absolute (DQN) and relative (CW-Q)
critics are compared on equal terms.  Each agent is scored on its own state distribution and its own policy pi.

usage: python rank_minatar.py --ckpt X.pkl --jsonl run.jsonl --out res.json [--n_states 256 --K 16 --H 500]
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/home/junweiluo/Stoix")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def build_q(cfg, A, kind):
    """-> q(params, obs[B, ...]) -> [B, A] action scores (DQN: Q; qcw: Borda/centred Q, same ordering)."""
    if kind == "pair":
        from stoix.systems.q_learning.ff_rpair import PairNet
        s = cfg["system"]
        head = str(s["head"])
        net = PairNet(num_actions=A, head=head, hidden=512 if head == "rc51" else 128,
                      emb_dim=48 if head == "rc51" else 16, rank=int(s.get("pair_rank", 32)),
                      n_atoms=int(s.get("rc_atoms", 51)), vmax=float(s.get("rc_vmax", 20.0)),
                      additive=bool(s.get("all_pairs", False)) and head in ("free", "anti"))
        return lambda p, o: net.apply(p, o, method=PairNet.scores)
    import hydra
    from omegaconf import OmegaConf
    from stoix.systems.q_learning.ff_rdqn import Actor
    ncfg = OmegaConf.create(cfg["network"])
    torso = hydra.utils.instantiate(ncfg.actor_network.pre_torso)
    head = hydra.utils.instantiate(ncfg.actor_network.action_head, action_dim=A, epsilon=0.0)
    net = Actor(torso=torso, action_head=head)
    return lambda p, o: net.apply(p["q"], o).preferences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--jsonl", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kind", required=True, choices=["dqn", "pair"])      # dqn: ff_rdqn checkpoint; pair: ff_rpair
    ap.add_argument("--n_states", type=int, default=256); ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--H", type=int, default=500); ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--B", type=int, default=32); ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=8192); ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    cfg = json.loads(open(a.jsonl).readline())["config"]
    game = cfg["env"]["scenario"]["name"]; gamma = float(cfg["system"]["gamma"])
    sticky = float(cfg["env"].get("kwargs", {}).get("sticky_action_prob", 0.1))
    import pgx
    env = pgx.make(game)
    if hasattr(env, "sticky_action_prob"):
        env.sticky_action_prob = sticky
    A = int(env.num_actions)
    params = pickle.load(open(a.ckpt, "rb"))
    params = jax.tree_util.tree_map(lambda x: x[0], params)        # drop the update_batch_size (=1) axis
    qf = build_q(cfg, A, a.kind)
    q = jax.jit(lambda o: qf(params, o.astype(jnp.float32)))
    step = jax.vmap(env.step)
    t0 = time.time()

    # ---- 1. states on the agent's own greedy trajectories
    key = jax.random.PRNGKey(a.seed)
    key, k0 = jax.random.split(key)
    s = jax.vmap(env.init)(jax.random.split(k0, a.B))

    @jax.jit
    def collect(s, key):
        def f(c, k):
            s, alive = c
            act = jnp.argmax(q(s.observation), -1)
            s2 = step(s, act, jax.random.split(k, a.B))
            alive2 = alive & ~(s2.terminated | s2.truncated)
            return (s2, alive2), (s, alive)
        _, (S, alive) = jax.lax.scan(f, (s, jnp.ones(a.B, bool)), jax.random.split(key, a.T))
        return S, alive
    key, k1 = jax.random.split(key)
    S, alive = collect(s, k1)
    alive = np.array(alive); alive[np.arange(a.T) % a.stride != 0] = False
    cand = np.argwhere(alive)                                         # (t, b)
    rng = np.random.default_rng(a.seed)
    pick = cand[rng.choice(len(cand), size=min(a.n_states, len(cand)), replace=False)]
    N = len(pick)
    St = jax.tree_util.tree_map(lambda x: x[pick[:, 0], pick[:, 1]], S)
    Qnet = np.asarray(q(St.observation))                              # [N, A]

    # ---- 2. Monte Carlo action values of the greedy policy
    M = N * A * a.K
    rep = lambda x: jnp.repeat(x, A * a.K, axis=0)  # noqa: E731
    S0 = jax.tree_util.tree_map(rep, St)
    a0 = jnp.tile(jnp.repeat(jnp.arange(A), a.K), N)

    @jax.jit
    def mc(S0, a0, key):
        n = a0.shape[0]
        k0, key = jax.random.split(key)
        s = step(S0, a0, jax.random.split(k0, n))
        G = s.rewards[:, 0].astype(jnp.float32)
        live = ~(s.terminated | s.truncated)

        def f(c, kk):
            s, G, live, disc = c
            act = jnp.argmax(q(s.observation), -1)
            s2 = step(s, act, jax.random.split(kk, n))
            G = G + disc * live * s2.rewards[:, 0].astype(jnp.float32)
            live = live & ~(s2.terminated | s2.truncated)
            return (s2, G, live, disc * gamma), None
        (s, G, live, _), _ = jax.lax.scan(f, (s, G, live, jnp.float32(gamma)), jax.random.split(key, a.H - 1))
        return G
    Gs = []
    for i in range(0, M, a.chunk):
        key, kk = jax.random.split(key)
        sl = lambda x: x[i:i + a.chunk]  # noqa: E731
        Gs.append(np.asarray(mc(jax.tree_util.tree_map(sl, S0), a0[i:i + a.chunk], kk)))
    G = np.concatenate(Gs).reshape(N, A, a.K)
    Qhat = G.mean(-1); se = G.std(-1, ddof=1) / np.sqrt(a.K)

    # ---- 3. metrics
    from scipy.stats import kendalltau
    taus, regrets, top1, n_sig, n_ok, n_const = [], [], [], 0, 0, 0
    for i in range(N):
        if np.ptp(Qhat[i]) < 1e-9:
            n_const += 1
        else:
            t = kendalltau(Qnet[i], Qhat[i]).statistic
            if np.isfinite(t):
                taus.append(t)
        g = int(np.argmax(Qnet[i])); b = int(np.argmax(Qhat[i]))
        r = Qhat[i, b] - Qhat[i, g]; regrets.append(r)
        top1.append(r <= 2 * np.sqrt(se[i, b] ** 2 + se[i, g] ** 2) + 1e-12)
        for x in range(A):
            for y in range(x + 1, A):
                d = Qhat[i, x] - Qhat[i, y]
                if abs(d) > 2 * np.sqrt(se[i, x] ** 2 + se[i, y] ** 2) and abs(d) > 1e-9:
                    n_sig += 1; n_ok += int(np.sign(Qnet[i, x] - Qnet[i, y]) == np.sign(d))
    res = {"ckpt": os.path.basename(a.ckpt), "game": game, "N": N, "A": A, "K": a.K, "H": a.H,
           "tau": float(np.mean(taus)) if taus else None, "n_tau": len(taus), "n_const": n_const,
           "sig_acc": n_ok / n_sig if n_sig else None, "n_sig_pairs": n_sig, "sig_pairs_per_state": n_sig / N,
           "regret": float(np.mean(regrets)), "top1": float(np.mean(top1)),
           "qhat_spread": float(np.mean(np.ptp(Qhat, 1))), "se_mean": float(se.mean()),
           "qnet_spread": float(np.mean(np.ptp(Qnet, 1))), "wall": time.time() - t0}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res))
    np.savez_compressed(a.out.replace(".json", ".npz"), Qnet=Qnet, Qhat=Qhat, se=se)


if __name__ == "__main__":
    main()
