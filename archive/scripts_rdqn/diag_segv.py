"""2026-11-05 minimal segfault repro: Playground env jit reset/step (vmap 128) and a 64-step scan, no Stoix."""
import sys, time, faulthandler; faulthandler.enable(all_threads=True)
import jax, jax.numpy as jnp, mujoco_playground as mp, mujoco, mujoco.mjx
print("jax", jax.__version__, "mujoco", mujoco.__version__, "playground", getattr(mp, "__version__", "?"), jax.devices(), flush=True)
name = sys.argv[1]; stage = sys.argv[2] if len(sys.argv) > 2 else "all"
env = mp.registry.load(name, config=mp.registry.get_default_config(name))
keys = jax.random.split(jax.random.PRNGKey(0), 128)
t = time.time(); st = jax.jit(jax.vmap(env.reset))(keys); jax.block_until_ready(st); print("reset ok", time.time() - t, flush=True)
a = jnp.zeros((128, env.action_size))
t = time.time(); st2 = jax.jit(jax.vmap(env.step))(st, a); jax.block_until_ready(st2); print("step ok", time.time() - t, flush=True)
def body(s, _): return jax.vmap(env.step)(s, a), None
t = time.time(); st3, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, 64))(st); jax.block_until_ready(st3); print("scan ok", time.time() - t, flush=True)
print("DONE", flush=True)
