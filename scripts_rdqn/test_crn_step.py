"""CRN mechanism check in Stoix/stoa gymnax (Asterix-MinAtar, full stoix wrapper stack).
1) stepping the SAME wrapped state twice with the same action is bit-identical (the step is a pure function);
2) the step noise comes from the key stored in the state: changing only that key changes the outcome of the
   same action at some states (so CRN = reuse the state; IND = would need a new key);
3) over many states, P(next obs identical | same action) = 1, and outcomes of a != b differ only where the
   action matters."""
import jax, jax.numpy as jnp, numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
from stoix.utils import make_env as environments

with initialize(version_base="1.2", config_path="../stoix/configs/default/anakin"):
    cfg = compose(config_name="default_ff_rdqn.yaml", overrides=["env=gymnax/asterix", "arch.total_num_envs=64"])
OmegaConf.set_struct(cfg, False)
cfg.num_devices = 1; cfg.arch.num_envs = 64
env, _ = environments.make(config=cfg)
keys = jax.random.split(jax.random.PRNGKey(0), 64)
st, ts = env.reset(keys)
step = jax.jit(env.step)
rng = np.random.default_rng(0)
same_ok, key_matters, n = True, 0, 0
for t in range(400):
    a = jnp.asarray(rng.integers(0, 5, 64), jnp.int32)
    s1, t1 = step(st, a); s2, t2 = step(st, a)
    same_ok &= bool(jnp.all(t1.extras["next_obs"] == t2.extras["next_obs"])) and bool(jnp.all(t1.reward == t2.reward))
    # perturb ONLY the adapter key inside the state (independent noise), same action
    leaves, treedef = jax.tree_util.tree_flatten_with_path(st)
    def perturb(path, x):
        return jax.random.split(jax.random.PRNGKey(t + 1000), x.shape[0]) if "rng_key" in jax.tree_util.keystr(path) else x
    st_ind = jax.tree_util.tree_unflatten(treedef, [perturb(p, x) for p, x in leaves])
    _, t3 = step(st_ind, a)
    diff = jnp.any(t3.extras["next_obs"].reshape(64, -1) != t1.extras["next_obs"].reshape(64, -1), axis=1)
    key_matters += int(diff.sum()); n += 64
    st, ts = s1, t1
print("same state + same action bit-identical:", same_ok)
print("fraction of steps where ONLY changing the state key changes the next obs: %.3f" % (key_matters / n))
print("key fields found:", [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(st)[0] if "rng_key" in jax.tree_util.keystr(p)])
