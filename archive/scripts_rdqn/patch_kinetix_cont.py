"""2026-11-08: cw_continuous.py support for pytree (Kinetix EntityObservation) observations + transformer critic.
system.obs_encoder = flat (default, old behaviour) | kinetix_tf.  Logs eval solve_rate when env.solved_return_threshold."""
import shutil
P = "stoix/systems/ddpg/cw_continuous.py"
s = open(P).read()
assert "obs_encoder" not in s, "already patched"
shutil.copy(P, P + ".bak_20261108_before_kinetix")
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:90]
    s = s.replace(a, b)
rep('''class PairCritic(nn.Module):''', '''class EntityQCritic(nn.Module):
    """2026-11-08 Kinetix: Q(s,a) = Dense1(MLP([KinetixTransformerEncoder(s), a]))."""
    hidden: Tuple[int, ...] = (128, 128, 128, 128, 128)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, obs, act):
        from stoix.networks.specialised.kinetix_tf import KinetixTransformerEncoder
        h = KinetixTransformerEncoder()(obs)
        return nn.Dense(1)(MLP(self.hidden, self.layer_norm)(jnp.concatenate([h, act], -1)))[..., 0]


def _nrows(tree):
    return jax.tree_util.tree_leaves(tree)[0].shape[0]


def _finite_rows(tree):
    ok = None
    for x in jax.tree_util.tree_leaves(tree):
        if jnp.issubdtype(x.dtype, jnp.floating):
            r = jnp.all(jnp.isfinite(x.reshape(x.shape[0], -1)), -1)
            ok = r if ok is None else (ok & r)
    return ok


def _zero_nonfinite(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.where(jnp.isfinite(x), x, 0.0) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


class PairCritic(nn.Module):''')
rep('''    config.system.action_minimum = float(env.action_space().minimum)
    config.system.action_maximum = float(env.action_space().maximum)''',
'''    _mn, _mx = np.asarray(env.action_space().minimum), np.asarray(env.action_space().maximum)
    assert np.all(_mn == _mn.flat[0]) and np.all(_mx == _mx.flat[0]), "per-dimension bounds must be equal"
    config.system.action_minimum = float(_mn.flat[0])
    config.system.action_maximum = float(_mx.flat[0])''')
rep('''    if head in ("q", "qcw"):
        net = QCritic(hidden=tuple(config.system.q_hidden), layer_norm=ln)''',
'''    obs_enc = str(config.system.get("obs_encoder", "flat"))
    if head in ("q", "qcw") and obs_enc == "kinetix_tf":
        net = EntityQCritic(hidden=tuple(config.system.q_hidden), layer_norm=ln)
    elif head in ("q", "qcw"):
        net = QCritic(hidden=tuple(config.system.q_hidden), layer_norm=ln)''')
rep('''    init_x = env.observation_space().generate_value()
    assert not isinstance(init_x, dict), "flat observation vectors only"
    init_x = init_x[None, ...]''',
'''    if obs_enc == "flat":
        init_x = env.observation_space().generate_value()
        assert not isinstance(init_x, dict), "flat observation vectors only"
        init_x = init_x[None, ...]
    else:                                   # pytree observation: take the structure from a real reset
        _, _ts0 = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
        init_x = _ts0.observation''')
rep('''            action = jax.random.uniform(k, (last_ts.observation.shape[0], action_dim), minval=a_min, maxval=a_max)''',
'''            action = jax.random.uniform(k, (_nrows(last_ts.observation), action_dim), minval=a_min, maxval=a_max)''')
rep('''        bad = ~(jnp.isfinite(ts.reward).reshape(-1) & jnp.all(jnp.isfinite(nobs.reshape(nobs.shape[0], -1)), -1))
        rew = jnp.where(jnp.isfinite(ts.reward), ts.reward, 0.0)          # 11-05: physics blow-up -> terminal, r = 0
        nobs = jnp.where(jnp.isfinite(nobs), nobs, 0.0)''',
'''        bad = ~(jnp.isfinite(ts.reward).reshape(-1) & _finite_rows(nobs))
        rew = jnp.where(jnp.isfinite(ts.reward), ts.reward, 0.0)          # 11-05: physics blow-up -> terminal, r = 0
        nobs = _zero_nonfinite(nobs)''')
rep('''    dummy = Transition(obs=init_x[0], action=jnp.zeros((action_dim,), dtype=float), reward=jnp.zeros((), dtype=float),
                       done=jnp.zeros((), dtype=bool), next_obs=init_x[0],''',
'''    _x0 = jax.tree_util.tree_map(lambda x: x[0], init_x)
    dummy = Transition(obs=_x0, action=jnp.zeros((action_dim,), dtype=float), reward=jnp.zeros((), dtype=float),
                       done=jnp.zeros((), dtype=bool), next_obs=_x0,''')
rep('''        rec["eval_length"] = _mean(evo.episode_metrics["episode_length"])''',
'''        rec["eval_length"] = _mean(evo.episode_metrics["episode_length"])
        if config.env.get("solved_return_threshold") is not None:          # 11-08 Kinetix holdout solve rate
            rec["eval_solve_rate"] = _mean(np.asarray(evo.episode_metrics["episode_return"])
                                           >= float(config.env.solved_return_threshold))''')
open(P, "w").write(s)
print("patched")
