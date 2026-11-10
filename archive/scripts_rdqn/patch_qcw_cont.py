"""2026-11-04: continuous CW-Q-rho (head=qcw) + community-alignment options for cw_continuous.py.
  head=qcw : unary critic Q(s,a) (QCritic), pair difference D(x,y) = Q(x) - Q(y); cross-weighted residual with the
             duel/all-pairs path (rho weight matrix, current-state parts from each critic's own TARGET), no scalar c.
             TD3: next-state value = min_k Q_k^-(s', a~) (standard clipped double Q), shared by both critics.
             Actor: -mean Q_1(s, mu(s)) (identical to TD3).
  system.activation : silu (default, old behaviour) | relu   (critic MLPs)
  system.warmup_random : False (default, old behaviour) | True -> uniform random actions during warmup (TD3/CleanRL)
run from /home/junweiluo/Stoix:  .venv/bin/python scripts_rdqn/patch_qcw_cont.py"""
import os, shutil
P = "stoix/systems/ddpg/cw_continuous.py"
BAK = P + ".bak_20261104_before_qcw"
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
s = open(BAK).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:80], n, count)
    s = s.replace(old, new)


rep("              = duel  CW-Duel: D((s,a),(t,b)) = W(s,t) + A(s,a) - A(t,b)\n",
    "              = duel  CW-Duel: D((s,a),(t,b)) = W(s,t) + A(s,a) - A(t,b)\n"
    "              = qcw   CW-Q (2026-11-04): D(x, y) = Q(x) - Q(y) with the plain unary critic Q(s,a); all-pairs /\n"
    "                      rho weight matrix as duel; TD3 next value min_k Q_k^-(s',a~); actor = TD3 actor. No scalar c.\n")
# activation option
rep("""class MLP(nn.Module):
    sizes: Tuple[int, ...]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for s in self.sizes:
            x = nn.Dense(s)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.silu(x)
        return x
""", """ACT = {"fn": nn.silu}   # set by build_system from system.activation (silu | relu)


class MLP(nn.Module):
    sizes: Tuple[int, ...]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for s in self.sizes:
            x = nn.Dense(s)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = ACT["fn"](x)
        return x
""")
# qcw critic loss
rep("""        assert head == "duel"
        W = pairs                                                         # [m, m] pair weights
""", """        if head == "qcw":
            W = pairs                                                     # [m, m] pair weights
            qn = jnp.min(jnp.stack([net.apply(tqp[c], tr.next_obs, a_n) for c in keys]), 0)
            r, d = tr.r, tr.d
            loss = 0.0
            for k_, c in enumerate(keys):
                q_on = net.apply(qp[c], tr.obs, tr.action)
                q_t = net.apply(tqp[c], tr.obs, tr.action)
                D = q_on[:, None] - q_on[None, :]
                M = qn[:, None] - qn[None, :]
                Amx = qn[:, None] - q_t[None, :]
                Bmn = q_t[:, None] - qn[None, :]
                xi = cross_xi(jax.lax.stop_gradient(D), M, Amx, Bmn, r[:, None], r[None, :], d[:, None], d[None, :],
                              gamma)
                T = jax.lax.stop_gradient(D + xi / (1.0 - gamma))
                loss = loss + 0.5 * jnp.sum(W * jnp.square(T - D))
                if k_ == 0:
                    u = r + gamma * (1.0 - d) * qn - q_on                     # DQN-style residual (diagnostic)
                    info.update({"q_mean": jnp.mean(q_on), "D_abs": jnp.sum(W * jnp.abs(D)),
                                 "td_std": jnp.sqrt(jnp.sum(W * jnp.square(T - D))),
                                 "u_mean": jnp.mean(u), "u_std": jnp.std(u)})
            return loss, info
        assert head == "duel"
        W = pairs                                                         # [m, m] pair weights
""")
rep("""        if head == "q":
            return jnp.mean(net.apply(qp0, obs, act_mu))
""", """        if head in ("q", "qcw"):
            return jnp.mean(net.apply(qp0, obs, act_mu))
""")
rep('assert head in ("q", "pair", "duel") and pairing in ("rho", "random")',
    'assert head in ("q", "pair", "duel", "qcw") and pairing in ("rho", "random")')
rep("""    ln = bool(config.system.get("layer_norm", False))
    if head == "q":
""", """    ln = bool(config.system.get("layer_norm", False))
    ACT["fn"] = {"silu": nn.silu, "relu": nn.relu}[str(config.system.get("activation", "silu"))]
    if head in ("q", "qcw"):
""")
# random warmup
rep("""    def env_step(actor_p, env_state, last_ts, k):
        action = behaviour(actor_p, last_ts.observation, k)
""", """    warm_rand = bool(config.system.get("warmup_random", False))

    def env_step(actor_p, env_state, last_ts, k, random_act=False):
        if random_act:
            action = jax.random.uniform(k, (last_ts.observation.shape[0], action_dim), minval=a_min, maxval=a_max)
        else:
            action = behaviour(actor_p, last_ts.observation, k)
""")
rep("""            env_state, ts, tr = env_step(params.actor_params.online, env_state, last_ts, kk)
            return (env_state, ts, k), tr""", """            env_state, ts, tr = env_step(params.actor_params.online, env_state, last_ts, kk, warm_rand)
            return (env_state, ts, k), tr""")
open(P, "w").write(s)
print("patched", P)
