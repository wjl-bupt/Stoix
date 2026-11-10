"""check_rho (2026-11-03): rho-kernel estimators.  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_rho.py"""
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.same_traj import rho_pairs, rdqn_rho_loss
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
rng = np.random.default_rng(0)
for t in range(6):
    G, ng = 8, int(rng.integers(3, 12)); m = G * ng
    idx = rng.integers(0, 40, m)                                   # small pool -> many coincident items
    val = rng.normal(size=40) * 2; u = val[idx] + 0.0         # residual is a function of the buffer item
    a, b = np.meshgrid(np.arange(G), np.arange(G), indexing="ij"); off = a != b
    ix_s = (np.arange(ng)[:, None] * G + a[off][None]).reshape(-1); iy_s = (np.arange(ng)[:, None] * G + b[off][None]).reshape(-1)
    ms = (idx[ix_s] != idx[iy_s]).astype(float); w_s = ms / max(ms.sum(), 1)
    rho = float(rng.uniform())
    L, Ls, Lc = rdqn_rho_loss(jnp.array(u), jnp.array(ix_s), jnp.array(iy_s), jnp.array(w_s), G, rho, jnp.array(idx))
    grp = np.arange(m) // G
    I, Jm = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    sel = (grp[I] != grp[Jm]) & (idx[I] != idx[Jm])
    Lc_bf = 0.25 * np.mean((u[I] - u[Jm])[sel] ** 2)
    Ls_bf = 0.25 * np.sum(w_s * (u[ix_s] - u[iy_s]) ** 2)
    check(f"RDQN cross-group closed form == brute force over distinct-item cross pairs (ng={ng})", abs(float(Lc) - Lc_bf) < 1e-5, (float(Lc), Lc_bf))
    check("RDQN L = (1-rho) L_same + rho L_cross", abs(float(L) - ((1 - rho) * Ls_bf + rho * Lc_bf)) < 1e-5)
    K = 7 if ng > 7 else ng - 1
    ix, iy, w = rho_pairs(jax.random.PRNGKey(t), jnp.array(idx), ng, G, K, rho, jnp.array(ix_s), jnp.array(iy_s), jnp.array(w_s))
    ix, iy, w = map(np.asarray, (ix, iy, w)); ns = len(ix_s)
    cx, cy, cw = ix[ns:], iy[ns:], w[ns:]
    check("CW cross pairs: different groups, K per slot", (grp[cx] != grp[cy]).all() and np.all(np.bincount(cx, minlength=m) == K))
    check("CW weights: same sums 1-rho, cross sums rho, self/coincident pairs weight 0",
          abs(w[:ns].sum() - (1 - rho)) < 1e-5 and abs(cw.sum() - rho) < 1e-5 and np.all(cw[idx[cx] == idx[cy]] == 0)
          and np.all(w[:ns][idx[ix_s] == idx[iy_s]] == 0))
print(f"{sum(ok)}/{len(ok)} passed")
