"""2026-11-05: sanitise MJX physics blow-ups in cw_continuous.env_step.  Playground DMC envs terminate (done=1) when
qpos/qvel become NaN, but the stored reward / next_obs of that transition are NaN and poison the critic (seen: WalkerRun
td3 s0 at 2.5M).  Non-finite reward -> 0, non-finite next_obs entries -> 0, and such a transition is marked terminal.
No-op (bitwise) when everything is finite."""
import os, shutil
P = "stoix/systems/ddpg/cw_continuous.py"; BAK = P + ".bak_20261105_before_nan"
if not os.path.exists(BAK): shutil.copy(P, BAK)
s = open(BAK).read()
old = """        done = (ts.terminated() if trunc_boot else ts.last()).reshape(-1)
        tr = Transition(last_ts.observation, action, ts.reward, done, ts.extras["next_obs"],
                        ts.extras["episode_metrics"])"""
new = """        done = (ts.terminated() if trunc_boot else ts.last()).reshape(-1)
        nobs = ts.extras["next_obs"]
        bad = ~(jnp.isfinite(ts.reward).reshape(-1) & jnp.all(jnp.isfinite(nobs.reshape(nobs.shape[0], -1)), -1))
        rew = jnp.where(jnp.isfinite(ts.reward), ts.reward, 0.0)          # 11-05: physics blow-up -> terminal, r = 0
        nobs = jnp.where(jnp.isfinite(nobs), nobs, 0.0)
        done = jnp.logical_or(done.astype(bool), bad).astype(done.dtype)
        tr = Transition(last_ts.observation, action, rew, done, nobs, ts.extras["episode_metrics"])"""
assert s.count(old) == 1; s = s.replace(old, new); open(P, "w").write(s); print("patched", P)
