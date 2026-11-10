"""2026-11-05: cheetah slowness fix (no version change).  Playground CheetahRun.reset runs 200 'stabilise' physics steps.
Stoa's AutoResetWrapper uses a per-env lax.cond inside vmap -> becomes a select -> reset (200 substeps) is computed for
every env at EVERY step (measured: 0.35 s / vector step vs 2.6 ms for env.step alone).
  make_env.py : env.batch_cond_reset=True -> BatchCondAutoResetVmap: vmapped step, then ONE lax.cond on any(done)
                over the batch (resets computed only on steps where some env ended; selects per env).  Same reset
                semantics/keys as AutoResetWrapper (key = split(state.rng_key)[0]); terminal obs kept in extras.
  cw_continuous.py : system.no_batch_vmap=True (requires update_batch_size 1) -> learner does not vmap over the
                update-batch axis, so the cond above stays a real cond (under vmap it would again become a select).
Defaults off -> bitwise old behaviour."""
import os, shutil
def patch(P, reps, tag):
    BAK = P + f".bak_20261105_before_{tag}"
    if not os.path.exists(BAK): shutil.copy(P, BAK)
    s = open(BAK).read()
    for item in reps:
        old, new = item[0], item[1]; first = len(item) > 2   # third element: replace the FIRST occurrence only
        assert (s.count(old) >= 1) if first else (s.count(old) == 1), (P, old[:70], s.count(old))
        s = s.replace(old, new, 1)
    open(P, "w").write(s); print("patched", P)

ME = "stoix/utils/make_env.py"
patch(ME, [
("""from stoix.utils.env_factory import EnvFactory""",
"""import jax.numpy as jnp
from stoa.core_wrappers.auto_reset import add_obs_to_extras
from stoa.core_wrappers.wrapper import Wrapper

from stoix.utils.env_factory import EnvFactory


class BatchCondAutoResetVmap(Wrapper):
    \"\"\"2026-11-05: vmapped env with auto-reset decided by ONE lax.cond on any(done) over the batch (see
    scripts_rdqn/patch_batchreset.py).  Equivalent to AutoResetWrapper + VmapWrapper, but the reset is only
    computed on steps where at least one env ended.\"\"\"

    def __init__(self, env):
        super().__init__(env)
        self._vreset = jax.vmap(self._env.reset, in_axes=(0, None))
        self._vstep = jax.vmap(self._env.step, in_axes=(0, 0, None))

    def reset(self, rng_key, env_params=None):
        state, ts = self._vreset(rng_key, env_params)
        return state, add_obs_to_extras(ts)

    def step(self, state, action, env_params=None):
        state, ts = self._vstep(state, action, env_params)
        ts = add_obs_to_extras(ts)
        done = ts.done()

        def _reset(args):
            st, t = args
            keys = jax.vmap(lambda k: jax.random.split(k)[0])(st.rng_key)
            rs, rts = self._vreset(keys, env_params)
            sel = lambda a, b: jnp.where(done.reshape(done.shape + (1,) * (a.ndim - done.ndim)), a, b)  # noqa
            return jax.tree_util.tree_map(sel, rs, st), t.replace(
                observation=jax.tree_util.tree_map(sel, rts.observation, t.observation))

        return jax.lax.cond(jnp.any(done), _reset, lambda a: a, (state, ts))"""),
("""    else:
        if config.env.get("use_cached_auto_reset", False):""",
"""    elif config.env.get("batch_cond_reset", False):   # 11-05 (apply_core_wrappers = first occurrence)
        env = BatchCondAutoResetVmap(env)
    else:
        if config.env.get("use_cached_auto_reset", False):""", "first"),
], "batchreset")

CW = "stoix/systems/ddpg/cw_continuous.py"
patch(CW, [
("""        a_grads, a_info = jax.lax.pmean(jax.lax.pmean((a_grads, a_info), "batch"), "device")
        q_grads, q_info = jax.lax.pmean(jax.lax.pmean((q_grads, q_info), "batch"), "device")""",
"""        a_grads, a_info = jax.lax.pmean(pmb((a_grads, a_info)), "device")
        q_grads, q_info = jax.lax.pmean(pmb((q_grads, q_info)), "device")"""),
("""    def learner_fn(ls):
        batched = jax.vmap(update_step, in_axes=(0, None), axis_name="batch")""",
"""    def learner_fn(ls):
        if NO_BV:   # 11-05: no vmap over the update-batch axis (size 1) -> env auto-reset cond stays a real cond
            ls0 = jax.tree_util.tree_map(lambda x: x[0], ls)
            ls0, (ep_info, loss_info) = jax.lax.scan(update_step, ls0, None, config.arch.num_updates_per_eval)
            ls = jax.tree_util.tree_map(lambda x: x[None], ls0)
            ep_info, loss_info = jax.tree_util.tree_map(lambda x: x[:, None], (ep_info, loss_info))
            return AnakinExperimentOutput(learner_state=ls, episode_metrics=ep_info, train_metrics=loss_info)
        batched = jax.vmap(update_step, in_axes=(0, None), axis_name="batch")"""),
("""    def update_epoch(carry, _):""",
"""    NO_BV = bool(config.system.get("no_batch_vmap", False))
    assert not NO_BV or int(config.arch.update_batch_size) == 1, "no_batch_vmap needs update_batch_size 1"
    pmb = (lambda x: x) if NO_BV else (lambda x: jax.lax.pmean(x, "batch"))

    def update_epoch(carry, _):"""),
], "batchreset")
