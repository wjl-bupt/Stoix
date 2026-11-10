"""C51 projection checks (float32 tolerances: support step 0.8, errors ~2e-5)."""
import sys; sys.path.insert(0, "/home/junweiluo/Stoix")
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rdqn import c51_project
ok = tot = 0
def chk(n, c, e=""):
    global ok, tot; tot += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + n, e)
rng = np.random.default_rng(0); N = 51; z = jnp.linspace(-20, 20, N)
p = jax.nn.softmax(jnp.array(rng.normal(size=(64, N))) * 2, -1)
r = jnp.array(rng.normal(size=64)); d = jnp.array((rng.random(64) > .2) * 0.99, jnp.float32)
t = c51_project(p, r, d, z)
chk("mass = 1", float(jnp.abs(t.sum(-1) - 1).max()) < 1e-4)
chk("non-negative", float(t.min()) >= -1e-7)
mean_t = (t * z).sum(-1); mean_src = (p * jnp.clip(r[:, None] + d[:, None] * z, -20, 20)).sum(-1)
chk("mean preserved (projection is mean-preserving inside support)", float(jnp.abs(mean_t - mean_src).max()) < 1e-3,
    f"{float(jnp.abs(mean_t - mean_src).max()):.2e}")
# terminal: delta at r
t2 = c51_project(p, jnp.full(64, 0.4), jnp.zeros(64), z)
chk("terminal target mean = r", float(jnp.abs((t2 * z).sum(-1) - 0.4).max()) < 1e-4)
# brute force projection (Bellemare Alg. 1)
def brute(pi, ri, di):
    m = np.zeros(N); dz = 40 / (N - 1)
    for j in range(N):
        tz = np.clip(ri + di * float(z[j]), -20, 20); b = (tz + 20) / dz; l, u = int(np.floor(b)), int(np.ceil(b))
        if l == u: m[l] += pi[j]
        else: m[l] += pi[j] * (u - b); m[u] += pi[j] * (b - l)
    return m
bm = np.stack([brute(np.asarray(p[i]), float(r[i]), float(d[i])) for i in range(64)])
chk("matches Bellemare Alg. 1", np.abs(bm - np.asarray(t)).max() < 1e-4, f"{np.abs(bm - np.asarray(t)).max():.2e}")
print(f"{ok}/{tot} passed")
