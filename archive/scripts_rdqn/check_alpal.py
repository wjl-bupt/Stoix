"""check_alpal (2026-11-03).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_alpal.py"""
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.alpal import al_penalty
ok = []
def check(n, c, i=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n, i)
rng = np.random.default_rng(0)
for t in range(5):
    B, A = 64, int(rng.integers(2, 7))
    vc, vn = rng.normal(size=(B, A)), rng.normal(size=(B, A)); a = rng.integers(0, A, B); d = rng.random(B) < 0.3
    G = vc.max(1) - vc[np.arange(B), a]; Gn = vn.max(1) - vn[np.arange(B), a]
    ref_al, ref_pal = G, np.minimum(G, (1 - d) * Gn)
    pa = np.asarray(al_penalty(jnp.array(vc), jnp.array(vn), jnp.array(a), jnp.array(d), "al"))
    pp = np.asarray(al_penalty(jnp.array(vc), jnp.array(vn), jnp.array(a), jnp.array(d), "pal"))
    check(f"penalty AL/PAL == numpy reference (A={A})", np.allclose(pa, ref_al, atol=1e-6) and np.allclose(pp, ref_pal, atol=1e-6))
    check("PAL <= AL, >= 0, PAL=0 on terminal", (pp <= pa + 1e-7).all() and (pp >= -1e-7).all() and np.allclose(pp[d], 0))
    # invariance to a constant shift (RDQN c) and to positive scaling sign
    pa2 = np.asarray(al_penalty(jnp.array(vc + 3.7), jnp.array(vn - 1.2), jnp.array(a), jnp.array(d), "pal"))
    check("penalty invariant to constant shift of Q", np.allclose(pp, pa2, atol=1e-5))
print(f"{sum(ok)}/{len(ok)} passed")
