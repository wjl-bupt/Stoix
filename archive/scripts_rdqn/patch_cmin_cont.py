"""2026-11-08: level-free (centered) clipped-double min for head=qcw in cw_continuous.py (system.min_mode=raw|centered).
critic k uses qn_k = min_j (Q_j^- - c_j) + c_k, c_j = batch mean of Q_j^-(s', a~).  Default raw = old behaviour."""
import shutil, sys
P = "stoix/systems/ddpg/cw_continuous.py"
s = open(P).read()
assert "min_mode" not in s, "already patched"
shutil.copy(P, P + ".bak_20261108_before_cmin")
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
rep("def make_critic_fns(net, head: str, gamma: float, n_critics: int):",
    "def make_critic_fns(net, head: str, gamma: float, n_critics: int, min_mode: str = \"raw\"):")
rep('''            qn = jnp.min(jnp.stack([net.apply(tqp[c], tr.next_obs, a_n) for c in keys]), 0)
            r, d = tr.r, tr.d
            loss = 0.0
            for k_, c in enumerate(keys):
                q_on = net.apply(qp[c], tr.obs, tr.action)''',
'''            qn_k = jnp.stack([net.apply(tqp[c], tr.next_obs, a_n) for c in keys])
            qn = jnp.min(qn_k, 0)
            if min_mode == "centered" and n_critics == 2:          # 2026-11-08 level-free min
                cmean = jnp.mean(qn_k, axis=1, keepdims=True)
                m_c = jnp.min(qn_k - cmean, 0)
                qn_own = [m_c + cmean[j, 0] for j in range(n_critics)]
            else:
                qn_own = [qn] * n_critics
            if n_critics == 2:
                cmean_ = jnp.mean(qn_k, axis=1, keepdims=True)
                info.update({"pick0": jnp.mean((qn_k[0] <= qn_k[1]).astype(jnp.float32)),
                             "pick0c": jnp.mean((qn_k[0] - cmean_[0] <= qn_k[1] - cmean_[1]).astype(jnp.float32)),
                             "lvl_gap": jnp.mean(qn_k[0] - qn_k[1])})
            r, d = tr.r, tr.d
            loss = 0.0
            for k_, c in enumerate(keys):
                qn = qn_own[k_]
                q_on = net.apply(qp[c], tr.obs, tr.action)''')
rep("    critic_loss, actor_value = make_critic_fns(net, head, gamma, n_critics)",
    "    critic_loss, actor_value = make_critic_fns(net, head, gamma, n_critics,\n"
    "                                               str(config.system.get(\"min_mode\", \"raw\")))")
open(P, "w").write(s)
print("patched")
