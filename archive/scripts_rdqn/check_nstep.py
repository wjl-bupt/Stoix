"""Unit checks for nstep_loss in ff_rdqn.py (2026-10-31)."""
import sys, types
sys.argv = sys.argv[:1]
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rdqn import nstep_loss
from typing import NamedTuple

class Tr(NamedTuple):
    obs: object; action: object; reward: object; done: object; next_obs: object

def net_apply(params, obs):
    return types.SimpleNamespace(preferences=params["tab"][obs])

rng = np.random.default_rng(0)
S, A, B, g = 9, 4, 64, 0.97
ok = 0; tot = 0
def chk(name, a, b, tol=1e-4):
    global ok, tot
    tot += 1; e = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    good = e < tol; ok += good
    print(f"{'PASS' if good else 'FAIL'} {name}: err {e:.2e}")

for n in [1, 2, 3, 5]:
    obs = rng.integers(0, S, size=(B, n)); nobs = rng.integers(0, S, size=(B, n))
    act = rng.integers(0, A, size=(B, n)); rew = rng.normal(size=(B, n)).astype(np.float32)
    don = rng.random((B, n)) < 0.25
    tr = Tr(jnp.array(obs), jnp.array(act), jnp.array(rew), jnp.array(don), jnp.array(nobs))
    q = rng.normal(size=(S, A)).astype(np.float32); qt = rng.normal(size=(S, A)).astype(np.float32)
    c, ct = np.float32(0.7), np.float32(-0.4)
    p = {"q": {"tab": jnp.array(q)}, "c": jnp.array(c)}; tp = {"q": {"tab": jnp.array(qt)}, "c": jnp.array(ct)}
    # naive
    yn = np.zeros(B); y1 = np.zeros(B); corr = np.zeros(B); Gam = np.zeros(B)
    for j in range(B):
        alive = 1.0; G = 0.0
        for i in range(n):
            G += g**i * alive * rew[j, i]
            if i >= 1:
                d1 = rew[j, i] + g * (1 - don[j, i]) * (qt[nobs[j, i]].max() + ct) - qt[obs[j, i], act[j, i]] - ct
                corr[j] += g**i * alive * d1
            alive *= (1 - don[j, i])
        Gam[j] = g**n * alive
        yn[j] = G + Gam[j] * (qt[nobs[j, n - 1]].max() + ct)
        y1[j] = rew[j, 0] + g * (1 - don[j, 0]) * (qt[nobs[j, 0]].max() + ct)
    qa = q[obs[:, 0], act[:, 0]]
    for mode in ["same", "sep_cor", "sep_unc"]:
        for rd in [False, True]:
            ynet = yn if mode == "same" else y1
            u = ynet - qa
            nl = 0.5 * np.mean((u - u.mean()) ** 2) if rd else 0.5 * np.mean((u - c) ** 2)
            dc = (y1 - qa + corr - c) if mode == "sep_cor" else (yn - qa - c)
            f = lambda pp: nstep_loss(pp, tp, tr, net_apply, g, n, mode, rd, False, 1000.0)
            (L, info), gr = jax.value_and_grad(f, has_aux=True)(p)
            chk(f"n={n} {mode} rdqn={rd} net_loss", info["q_loss"], nl)
            chk(f"n={n} {mode} rdqn={rd} mean_delta_c", info["mean_delta"], dc.mean())
            chk(f"n={n} {mode} rdqn={rd} grad_c=-2 mean delta_c", gr["c"], -2 * dc.mean())
            chk(f"n={n} {mode} rdqn={rd} Gamma_n", info["gamma_n_mean"], Gam.mean())
    # level coefficient: joint shift of q and q- by b changes scalar residual by -b(1 - Gamma_n)
    bsh = 2.3
    for mode in ["same", "sep_cor"]:
        f0 = nstep_loss(p, tp, tr, net_apply, g, n, mode, True, False, 1000.0)[1]["mean_delta"]
        p2 = {"q": {"tab": p["q"]["tab"] + bsh}, "c": p["c"]}; tp2 = {"q": {"tab": tp["q"]["tab"] + bsh}, "c": tp["c"]}
        f1 = nstep_loss(p2, tp2, tr, net_apply, g, n, mode, True, False, 1000.0)[1]["mean_delta"]
        chk(f"n={n} {mode} level coeff -(1-Gamma_n)", (f1 - f0) / bsh, -(1 - Gam).mean())
    # n=1: all modes coincide with the 1-step item-buffer target
    if n == 1:
        chk("n=1 y_n == y_1", yn, y1, 1e-5)
print(f"{ok}/{tot} passed")
