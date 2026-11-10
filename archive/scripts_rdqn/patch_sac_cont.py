"""2026-11-06: SAC base for cw_continuous.py (baseline only: head=q), aligned to the MuJoCo Playground Brax SAC config.
  actor  : MLP torso (config.network.actor_network.pre_torso) + Stoix NormalAffineTanhDistributionHead (min scale 1e-3)
  critics: two QCritic (same as TD3 arm, LayerNorm per system.layer_norm); target = r + g(1-d)(min_k Q_k^-(s',a') -
           alpha logpi(a'|s')),  a' ~ pi(.|s') (online actor, as Brax/CleanRL)
  actor  : mean(alpha logpi(a|s) - min_k Q_k(s,a)), a ~ pi reparameterised; updated every step
  alpha  : log_alpha stored as q_params["log_alpha"] (shares the critic Adam; critic loss uses sg(alpha), so it only
           gets the alpha-loss gradient); alpha loss = mean(alpha * sg(-logpi - target_entropy)),
           target_entropy = -system.target_entropy_scale * |A| (Brax: 0.5), init alpha = system.init_alpha (Brax: 1)
  behaviour: sample from pi (no extra noise); evaluation: use arch.evaluation_greedy=True (tanh(mean)).
run from /home/junweiluo/Stoix: .venv/bin/python scripts_rdqn/patch_sac_cont.py"""
import os, shutil
P = "stoix/systems/ddpg/cw_continuous.py"; BAK = P + ".bak_20261106_before_sac"
if not os.path.exists(BAK): shutil.copy(P, BAK)
s = open(BAK).read()
def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (old[:80], s.count(old)); s = s.replace(old, new)

rep("""    assert base in ("ddpg", "td3")
    td3 = base == "td3\"""", """    assert base in ("ddpg", "td3", "sac")
    td3 = base == "td3"
    sac = base == "sac\"""")
rep("""    n_critics = 2 if td3 else 1""", """    n_critics = 2 if (td3 or sac) else 1
    assert not sac or head == "q", "SAC base is a baseline only (head=q)\"""")
rep("""    actor_network = Actor(torso=actor_torso, action_head=CompositeNetwork([actor_head, post]))
    mu = lambda p, o: actor_network.apply(p, o).mode().clip(a_min, a_max)  # noqa: E731""",
"""    if sac:   # 11-06: stochastic tanh-Gaussian actor (Stoix SAC head)
        from stoix.networks.heads import NormalAffineTanhDistributionHead
        actor_network = Actor(torso=actor_torso, action_head=NormalAffineTanhDistributionHead(
            action_dim=action_dim, minimum=a_min, maximum=a_max))
    else:
        actor_network = Actor(torso=actor_torso, action_head=CompositeNetwork([actor_head, post]))
    mu = lambda p, o: actor_network.apply(p, o).mode().clip(a_min, a_max)  # noqa: E731""")
rep("""    q_params = {f"c{k}": net.init(q_keys[k], init_x, init_a) for k in range(n_critics)}
""", """    q_params = {f"c{k}": net.init(q_keys[k], init_x, init_a) for k in range(n_critics)}
    if sac:
        q_params["log_alpha"] = jnp.log(jnp.asarray(float(config.system.get("init_alpha", 1.0)), jnp.float32))
        target_entropy = -float(config.system.get("target_entropy_scale", 0.5)) * action_dim
""")
rep("""    if td3:
        actor_optim = optax.conditionally_mask(""", """    if td3 and not sac:
        actor_optim = optax.conditionally_mask(""")
rep("""    def behaviour(actor_p, obs, k):
        a = mu(actor_p, obs)""", """    def behaviour(actor_p, obs, k):
        if sac:
            return actor_network.apply(actor_p, obs).sample(seed=k).clip(a_min, a_max)
        a = mu(actor_p, obs)""")
rep("""    def update_epoch(carry, _):
        params, opt_states, buffer_state, key = carry""", """    def sac_epoch(params, opt_states, tr, kn):
        k1, k2 = jax.random.split(kn)
        keys_c = [f"c{k}" for k in range(n_critics)]
        actor_on = params.actor_params.online

        def q_side_loss(qp):
            alpha = jnp.exp(qp["log_alpha"]); alpha_sg = jax.lax.stop_gradient(alpha)
            pi_n = actor_network.apply(actor_on, tr.next_obs)
            a_n = pi_n.sample(seed=k1); logp_n = pi_n.log_prob(a_n)
            q_n = jnp.min(jnp.stack([net.apply(params.q_params.target[c], tr.next_obs, a_n) for c in keys_c]), 0)
            y = jax.lax.stop_gradient(tr.r + gamma * (1.0 - tr.d) * (q_n - alpha_sg * logp_n))
            qs = [net.apply(qp[c], tr.obs, tr.action) for c in keys_c]
            closs = sum(0.5 * jnp.mean(jnp.square(y - q)) for q in qs)
            pi = actor_network.apply(actor_on, tr.obs)
            logp = jax.lax.stop_gradient(pi.log_prob(pi.sample(seed=k2)))
            aloss = jnp.mean(alpha * jax.lax.stop_gradient(-logp - target_entropy))
            return closs + aloss, {"q_mean": jnp.mean(qs[0]), "td_std": jnp.std(y - qs[0]), "alpha": alpha,
                                   "alpha_loss": aloss}

        def actor_loss_fn(ap):
            alpha = jax.lax.stop_gradient(jnp.exp(params.q_params.online["log_alpha"]))
            pi = actor_network.apply(ap, tr.obs)
            a = pi.sample(seed=k2); logp = pi.log_prob(a)
            q = jnp.min(jnp.stack([net.apply(params.q_params.online[c], tr.obs, a) for c in keys_c]), 0)
            loss = jnp.mean(alpha * logp - q)
            return loss, {"actor_loss": loss, "entropy": -jnp.mean(logp)}

        a_grads, a_info = jax.grad(actor_loss_fn, has_aux=True)(actor_on)
        q_grads, q_info = jax.grad(q_side_loss, has_aux=True)(params.q_params.online)
        return a_grads, a_info, q_grads, q_info

    def update_epoch(carry, _):
        params, opt_states, buffer_state, key = carry""")
rep("""        a_grads, a_info = jax.grad(actor_loss_fn, has_aux=True)(params.actor_params.online)
        q_grads, q_info = jax.grad(critic_loss, has_aux=True)(
            params.q_params.online, params.q_params.target, tr, a_ref, a_n, a_n_ref, pairs)""",
"""        if sac:
            a_grads, a_info, q_grads, q_info = sac_epoch(params, opt_states, tr, kn)
        else:
            a_grads, a_info = jax.grad(actor_loss_fn, has_aux=True)(params.actor_params.online)
            q_grads, q_info = jax.grad(critic_loss, has_aux=True)(
                params.q_params.online, params.q_params.target, tr, a_ref, a_n, a_n_ref, pairs)""")
open(P, "w").write(s); print("patched", P)
