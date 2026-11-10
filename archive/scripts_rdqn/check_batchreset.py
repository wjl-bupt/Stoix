"""2026-11-05: BatchCondAutoResetVmap == AutoResetWrapper+VmapWrapper (CPU, CartpoleBalance + CheetahRun, 1100 steps,
episodes end at 1000 -> covers the reset).  JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_batchreset.py"""
import numpy as np, jax, jax.numpy as jnp
from omegaconf import OmegaConf
from stoix.utils import make_env as M
ok = []
import sys
for scen in sys.argv[1:] or ["CartpoleBalance", "CheetahRun"]:
    outs = []
    for flag in (False, True):
        cfg = OmegaConf.create({"env": {"env_name": "mujoco_playground", "scenario": {"name": scen, "task_name": scen},
                                        "kwargs": {}, "batch_cond_reset": flag},
                                "arch": {"num_envs": 8, "total_num_envs": 8}, "system": {}})
        env, _ = M.make_playground_env(scen, cfg)
        st, ts = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 8))
        step = jax.jit(env.step); rng = np.random.default_rng(0); obs, rew, dn, nobs = [], [], [], []
        for t in range(1003):
            a = jnp.array(rng.uniform(-1, 1, (8,) + env.action_space().shape), jnp.float32)
            st, ts = step(st, a)
            if t in (0, 1, 500, 998, 999, 1000, 1001, 1002):
                obs.append(np.asarray(ts.observation)); rew.append(np.asarray(ts.reward)); dn.append(np.asarray(ts.done()))
                nobs.append(np.asarray(ts.extras["next_obs"]))
        outs.append((obs, rew, dn, nobs))
    (o1, r1, d1, n1), (o2, r2, d2, n2) = outs
    for i, t in enumerate((0, 1, 500, 998, 999, 1000, 1001, 1002)):
        print(f"   t={t}: max|dobs| {np.abs(o1[i]-o2[i]).max():.2e}  max|dnext_obs| {np.abs(n1[i]-n2[i]).max():.2e}  max|drew| {np.abs(r1[i]-r2[i]).max():.2e}", flush=True)
    same = all(np.allclose(a, b, atol=1e-5) for a, b in zip(o1 + r1 + n1, o2 + r2 + n2)) and all((a == b).all() for a, b in zip(d1, d2))
    print(scen, "done flags at t=999,1000:", d1[4].any(), d1[5].any(), "| identical:", same, flush=True); ok.append(same)
print(f"{sum(ok)}/{len(ok)} passed")
