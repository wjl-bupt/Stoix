"""E1 check (2026-11-10): what does the same-trajectory level estimate of a TERMINAL sample carry?
CW-rho gradient on a terminal sample j (v_j = 1): u_j - kappa_j, kappa_j = sum_k w_jk u_k / sum_k w_jk v_k (partners k).
Its same-trajectory part (partners = other items of j's episode, all non-terminal, v = 1-g):
    kappa_same(T) = mean_{t in T, t != j} u_t / (1-g)           (exactly, when partners cover the episode)
Telescoping  sum_t u_t = sum_t r_t + g sum_t Q'(s_{t+1}) - sum_t Q(s_t,a_t)  shows kappa_same mixes the episode's
realised return with value estimates.  E1 hypothesis (11-09): "the episode outcome is written into the terminal target".
Measured per episode (own eps-greedy policy, eps .1, sticky .1, 128 envs, as in d1_kappa_noise.py; target net = online):
  ks      = kappa_same over the whole episode (expected over partner draws)
  ks7     = kappa_same from 7 random partners (what one batch sees)
  kg      = global level  mean(u)/mean(v) over the window (what cross partners estimate)
  G       = undiscounted episode return,  n = length,  Q0 = Q(s_0,a_0),  Gd = discounted return from s_0,
  mce     = Gd - Q0 (MC error of the start value),  qbar = mean Q(s_t,a_t) over the episode,
  qend    = mean max_a Q over the last 10 states (value at stake when dying),  u_term = u of the terminal sample.
Reports correlations of (ks - kg) with each, plus an OLS decomposition, and the size of ks - kg relative to the
terminal residual sd.  usage: e1_kappa_outcome.py --ckpt X.pkl --jsonl run.jsonl --out res.json"""
import argparse, json, os, pickle, sys, time
import numpy as np
sys.path.insert(0, "/home/junweiluo/Stoix"); sys.path.insert(0, "/home/junweiluo/Stoix/scripts_rdqn/ranking")
import jax, jax.numpy as jnp  # noqa: E402
from rank_minatar import build_q  # noqa: E402
import pgx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--jsonl", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--E", type=int, default=128); ap.add_argument("--T", type=int, default=3000); ap.add_argument("--burn", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.1); ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(); t0 = time.time()
    cfg = json.loads(open(a.jsonl).readline())["config"]
    game = cfg["env"]["scenario"]["name"]; g = float(cfg["system"]["gamma"])
    env = pgx.make(game); env.sticky_action_prob = 0.1; A = int(env.num_actions)
    params = jax.tree_util.tree_map(lambda x: x[0], pickle.load(open(a.ckpt, "rb")))
    qf = build_q(cfg, A, "pair"); q = jax.jit(lambda o: qf(params, o.astype(jnp.float32)) * (A - 1) / A)  # Borda -> Q units (qcw: = Q)
    step = jax.vmap(env.step); init = jax.vmap(env.init); E = a.E

    @jax.jit
    def run(key):
        k0, key = jax.random.split(key)
        s = init(jax.random.split(k0, E))
        def f(s, k):
            k1, k2, k3, k4 = jax.random.split(k, 4)
            qs = q(s.observation)
            act = jnp.where(jax.random.uniform(k1, (E,)) < a.eps, jax.random.randint(k2, (E,), 0, A), jnp.argmax(qs, -1))
            s2 = step(s, act, jax.random.split(k3, E))
            d = s2.terminated | s2.truncated
            out = (qs[jnp.arange(E), act], s2.rewards[:, 0], s2.terminated.astype(jnp.float32), d, jnp.max(q(s2.observation), -1))
            fresh = init(jax.random.split(k4, E))
            s3 = jax.tree_util.tree_map(lambda f_, o_: jnp.where(d.reshape((-1,) + (1,) * (o_.ndim - 1)), f_, o_), fresh, s2)
            return s3, out
        _, out = jax.lax.scan(f, s, jax.random.split(key, a.burn + a.T))
        return out
    qsa, r, term, done, qn = [np.asarray(x)[a.burn:].T for x in run(jax.random.PRNGKey(a.seed))]     # [E, T]
    v = 1 - g * (1 - term); u = r + g * (1 - term) * qn - qsa
    kg = u.mean() / v.mean()
    rng = np.random.default_rng(a.seed)
    rows = []
    for e in range(E):
        ends = np.where(done[e])[0]
        st = 0
        for en in ends:
            # complete episode = [st, en], but the first episode of the row may have started before the window (burn-in)
            if st > 0 and term[e, en] > 0 and en - st >= 2:
                idx = np.arange(st, en)                                   # non-terminal items of the episode
                uu = u[e, idx]
                ks = uu.sum() / ((1 - g) * len(idx))
                p7 = rng.choice(idx, 7, replace=len(idx) < 7)
                ks7 = u[e, p7].mean() / (1 - g)
                rr = r[e, st:en + 1]
                rows.append(dict(ks=ks, ks7=ks7, G=rr.sum(), n=en - st + 1, Q0=qsa[e, st], Gd=float((g ** np.arange(len(rr)) * rr).sum()),
                                 qbar=qsa[e, st:en + 1].mean(), qend=qn[e, max(st, en - 10):en].mean(), u_term=u[e, en],
                                 sum_r=rr.sum()))
            st = en + 1
    D = {k: np.array([x[k] for x in rows]) for k in rows[0]} if rows else {}
    res = {"ckpt": os.path.basename(a.ckpt), "game": game, "n_ep": len(rows), "kg": float(kg), "wall": time.time() - t0}
    if rows:
        D["mce"] = D["Gd"] - D["Q0"]
        dk = D["ks"] - kg
        cor = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
        res.update({"ks_minus_kg_mean": float(dk.mean()), "ks_minus_kg_sd": float(dk.std()), "ks7_minus_kg_sd": float((D["ks7"] - kg).std()),
                    "u_term_sd": float(D["u_term"].std()), "G_mean": float(D["G"].mean()), "n_mean": float(D["n"].mean()),
                    "corr_ks_ks7": cor(D["ks"], D["ks7"])})
        for k in ["G", "n", "Q0", "Gd", "mce", "qbar", "qend", "u_term"]:
            res[f"corr_ks_{k}"] = cor(dk, D[k]); res[f"corr_ks7_{k}"] = cor(D["ks7"] - kg, D[k])
        # telescoping predictor: (sum r - Q0) / (n (1-g)) - qbar   (approximate closed form of ks)
        pred = (D["sum_r"] - D["Q0"]) / (D["n"] * (1 - g)) - D["qbar"]
        res["corr_ks_telescope"] = cor(D["ks"], pred)
        # OLS of dk on standardised [G/n (reward rate), mce/n, qbar, qend]
        X = np.stack([D["G"] / D["n"], D["mce"] / D["n"], D["qbar"], D["qend"], 1 / D["n"]], 1)
        Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
        beta, *_ = np.linalg.lstsq(np.c_[Xs, np.ones(len(dk))], dk, rcond=None)
        pr = np.c_[Xs, np.ones(len(dk))] @ beta
        res["ols_names"] = ["rate=G/n", "mce/n", "qbar", "qend", "1/n"]; res["ols_beta"] = [float(b) for b in beta[:-1]]
        res["ols_r2"] = float(1 - ((dk - pr) ** 2).sum() / ((dk - dk.mean()) ** 2).sum())
        np.savez(a.out.replace(".json", ".npz"), **D, kg=kg)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w")); print(json.dumps(res))


if __name__ == "__main__":
    main()
