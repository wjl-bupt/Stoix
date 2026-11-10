"""2026-11-09 unit checks for same_traj.rho_weight_matrix_td."""
import sys
sys.path.insert(0, "/home/junweiluo/Stoix")
import numpy as np
import jax, jax.numpy as jnp
from stoix.systems.q_learning.same_traj import rho_weight_matrix, rho_weight_matrix_td
ok = 0; n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
rng = np.random.default_rng(0)
for trial in range(4):
    G, m = 8, 512
    idx = jnp.asarray(rng.integers(0, 3000, m))
    idx = idx.at[1].set(idx[0])                                    # a repeated buffer item inside group 0
    done = jnp.asarray(rng.random(m) < [0.0, 0.01, 0.05, 0.3][trial])
    W0 = np.asarray(rho_weight_matrix(idx, G, 0.5)); W, rem = rho_weight_matrix_td(idx, G, 0.5, done); W = np.asarray(W)
    grp = np.arange(m) // G; same = grp[:, None] == grp[None, :]; d = np.asarray(done)
    rm = same & (d[:, None] | d[None, :])
    chk(np.all(np.isfinite(W)) and np.all(W >= 0), f"[{trial}] finite, non-negative")
    chk(np.abs(W[rm]).max() == 0 if rm.any() else True, f"[{trial}] same-group pairs touching a terminal item have 0 weight")
    chk(np.allclose(W.sum(1), W0.sum(1), atol=1e-9), f"[{trial}] row sums preserved")
    chk(abs(W.sum() - 1) < 1e-5, f"[{trial}] total weight 1")
    chk(np.allclose(W[same & ~rm], W0[same & ~rm]), f"[{trial}] untouched same-group pairs unchanged")
    chk(np.all(np.diag(W) == 0) and W[0, 1] == 0, f"[{trial}] self / same-item pairs stay 0")
    chk(abs(float(rem) - W0[rm].sum()) < 1e-6, f"[{trial}] removed total reported ({float(rem):.4f})")
    if trial == 0:
        chk(np.allclose(W, W0), "[0] no terminal -> identical to rho_weight_matrix")
print(f"{ok}/{n} passed")
