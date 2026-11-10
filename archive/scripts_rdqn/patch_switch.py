"""2026-11-04 breakout diagnosis: loss switch on the CW-Duel (head=duel) path.
system.dqn_switch_at = T (env steps, default -1 = off), system.dqn_switch_dir = cw2dqn | dqn2cw
  cw2dqn: cross-weighted rho pair loss before T, DQN TD loss after T;  dqn2cw: the reverse.
  T = 0 with cw2dqn = DQN loss from the start on the same network (pair_rank 0 -> Q = w(s) + A(s,a) = Dueling-DQN head).
DQN loss: 0.5 mean (y - Q(s,a))^2, Q = w + A (+ low-rank term absent for rank 0), y = r + g (1-d) max_b Q-(s',b)
(target network), the same loss as ff_rdqn DQN with c = 0.  Same batch (rho groups) in both phases.
run from /home/junweiluo/Stoix:  .venv/bin/python scripts_rdqn/patch_switch.py"""
import shutil, os
P = "stoix/systems/q_learning/ff_rpair.py"
BAK = P + ".bak_20261104_before_switch"
if not os.path.exists(BAK):
    shutil.copy(P, BAK)
s = open(BAK).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (old[:70], n, count)
    s = s.replace(old, new)


rep("    def wloss(p, tp, tr, ix, iy, pw):\n        loss, info = compute_fn(p, tp, tr, ix, iy, pw=pw)\n        return loss, info\n",
    "    sw_at = int(config.system.get(\"dqn_switch_at\", -1)); sw_dir = str(config.system.get(\"dqn_switch_dir\", \"cw2dqn\"))\n"
    "    gamma_ = float(config.system.gamma); mar_ = float(config.system.max_abs_reward)\n"
    "    if sw_at >= 0:\n"
    "        assert str(config.system.head) == \"duel\" and int(config.system.get(\"pair_rank\", 32)) == 0 and sw_dir in (\"cw2dqn\", \"dqn2cw\")\n\n"
    "    def dqn_loss(p, tp, tr):\n"
    "        return duel_dqn_loss(net, p, tp, tr, gamma_, mar_)\n\n"
    "    def wloss(p, tp, tr, ix, iy, pw, t=None):\n"
    "        loss, info = compute_fn(p, tp, tr, ix, iy, pw=pw)\n"
    "        if sw_at >= 0 and t is not None:     # 2026-11-04 breakout diagnosis: loss switch at env step sw_at\n"
    "            l_dqn = dqn_loss(p, tp, tr)\n"
    "            after = t >= sw_at\n"
    "            use_dqn = after if sw_dir == \"cw2dqn\" else jnp.logical_not(after)\n"
    "            info = {**info, \"cw_loss\": loss, \"dqn_loss\": l_dqn, \"dqn_active\": use_dqn.astype(jnp.float32)}\n"
    "            loss = jnp.where(use_dqn, l_dqn, loss)\n"
    "        return loss, info\n")
rep("                    ix, iy, pw = rho_pairs(sk2, idx_, m // st_group, st_group, rho_kc, pair_rho, ix, iy, pw)\n"
    "                grads, loss_info = jax.grad(wloss, has_aux=True)(params.online, params.target, transitions, ix, iy, pw)\n",
    "                    ix, iy, pw = rho_pairs(sk2, idx_, m // st_group, st_group, rho_kc, pair_rho, ix, iy, pw)\n"
    "                grads, loss_info = jax.grad(wloss, has_aux=True)(params.online, params.target, transitions, ix, iy, pw,\n"
    "                                                                 opt_states[\"t\"])\n")
rep("def get_learner_fn(env, net, q_apply, q_update_fn, buffer_fns, config):\n",
    "def duel_dqn_loss(net, p, tp, tr, gamma, max_abs_reward):\n"
    "    \"\"\"2026-11-04: DQN TD loss on the head=duel network, Q = w(s) + A(s,a) (+ nothing else for rank 0); target net, no c.\"\"\"\n"
    "    mm = tr.action.shape[0]\n"
    "    w, _, _, Ac = net.apply(p, tr.obs, method=PairNet.duel_comp_obs)\n"
    "    wn, _, _, Acn = net.apply(tp, tr.next_obs, method=PairNet.duel_comp_obs)\n"
    "    q = w + Ac[jnp.arange(mm), tr.action]\n"
    "    r = jnp.clip(tr.reward, -max_abs_reward, max_abs_reward).astype(jnp.float32); d = tr.done.astype(jnp.float32)\n"
    "    y = jax.lax.stop_gradient(r + gamma * (1.0 - d) * (wn + Acn.max(-1)))\n"
    "    return 0.5 * jnp.mean(jnp.square(y - q))\n\n\n"
    "def get_learner_fn(env, net, q_apply, q_update_fn, buffer_fns, config):\n")
open(P, "w").write(s)
print("patched", P)
