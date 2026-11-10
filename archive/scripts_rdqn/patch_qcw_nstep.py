# 2026-11-09 (user): n-step CW-Q-rho (all-pairs qcw/duel heads, same_traj + pair_rho path).
# Every sampled item i gets its n-step tail from the flat item buffer (same env = stride E in write order):
#   R_i = sum_{k<n_i} g^k r_{t+k} (each step reward clipped), stop at the first stored done (episode end) or at the
#   newest stored step of that env;  Gam_i = g^{n_i} (1 - d_end)  (0 if the episode ended inside the window),
#   next state s_{t+n_i}.  The CW residual is the 1-step one with per-item discount Gam_i (v_i = 1 - Gam_i), i.e.
#   CW-Q-rho applied to the n-step look-ahead operator: xi_ij = v_j u_i - v_i u_j (+ the same target-lag terms),
#   k = (1 - g^n)/g^n, delta = xi / (1 - g^n).  Gauge-neutral for any per-item Gam (u_i -> u_i - c v_i).
# Uncorrected n-step returns (as n-step DQN / Rainbow).  n_step=1: unchanged code path.
import shutil
D='/home/junweiluo/Stoix/stoix/systems/q_learning/'
for f in ['same_traj.py','ff_rpair.py']:
    shutil.copy(D+f, D+f+'.bak_20261109_before_nstep')
s=open(D+'same_traj.py').read()
assert 'def nstep_tail' not in s
s+='''

def nstep_tail(buffer_state, idx, E: int, n: int, gamma: float, max_abs_reward: float):
    """2026-11-09 n-step tail of the items at ring positions idx (flat item buffer, rollout_length=1, E envs).
    Returns R [m] (discounted sum of clipped rewards), Gam [m] (gamma^{n_i} (1 - done at the last used step)),
    last [m] (ring position of the last used step, its next_obs is s_{t+n_i}), n_used [m]."""
    exp = jax.tree_util.tree_map(lambda x: x[0], buffer_state.experience)
    L = exp.done.shape[0]
    p = buffer_state.current_index
    o = jnp.mod(p - idx, L); o = jnp.where(o == 0, L, o)                   # offset back from the write pointer (1 = newest)
    rew = jnp.clip(exp.reward.astype(jnp.float32), -max_abs_reward, max_abs_reward)
    don = exp.done.astype(jnp.float32)
    R = rew[idx]; alive = 1.0 - don[idx]; Gam = gamma * alive; last = idx; nu = jnp.ones_like(R)
    for k in range(1, n):
        ok = alive * (o - k * E >= 1).astype(jnp.float32)                  # same env, k steps later, already stored
        pos = jnp.mod(idx + k * E, L)
        R = R + ok * (gamma ** k) * rew[pos]
        Gam = jnp.where(ok > 0, (gamma ** (k + 1)) * (1.0 - don[pos]), Gam)
        last = jnp.where(ok > 0, pos, last)
        nu = nu + ok
        alive = ok * (1.0 - don[pos])
    return R, Gam, last, nu
'''
open(D+'same_traj.py','w').write(s)

f=open(D+'ff_rpair.py').read()
def rep(o,n_,cnt=1):
    global f
    assert f.count(o)==cnt, (o[:70], f.count(o)); f=f.replace(o,n_)
rep('from stoix.systems.q_learning.same_traj import same_traj_batch, rho_pairs, rho_weight_matrix, rho_weight_matrix_td, balance_kernel, rho_weight_matrix_sparse',
    'from stoix.systems.q_learning.same_traj import same_traj_batch, rho_pairs, rho_weight_matrix, rho_weight_matrix_td, balance_kernel, rho_weight_matrix_sparse, nstep_tail')
rep('''                   cf_same: bool = False, cf_mixed: bool = True):
    k = (1.0 - gamma) / gamma''',
'''                   cf_same: bool = False, cf_mixed: bool = True, n_step: int = 1):
    k = (1.0 - gamma) / gamma
    gn = gamma ** n_step                                                # 2026-11-09 n-step: per-step discount of the look-ahead
    kn = (1.0 - gn) / gn''')
rep('''    def duel_all_pairs(p, tp, obs, act, nxt, r, d, pmat=None):''',
    '''    def duel_all_pairs(p, tp, obs, act, nxt, r, d, pmat=None, g_item=None):''')
rep('''        Gx, Gy = (gamma * (1 - d))[:, None], (gamma * (1 - d))[None, :]
        vx, vy = 1 - Gx, 1 - Gy
        W1 = k * Gx * Gy; W2 = Gx * (vy - k * Gy); W3 = Gy * (vx - k * Gx); W4 = -vy - Gy * (vx - k * Gx)
        Dsg = jax.lax.stop_gradient(Dm)
        xi = vy * r[:, None] - vx * r[None, :] + W1 * M + W2 * Amx + W3 * Bmn + W4 * Dsg
        delta = jax.lax.stop_gradient(xi) / (1.0 - gamma)
        mask = 1.0 - jnp.eye(m)
        npair = m * (m - 1)
        if pmat is None:
            loss = 0.5 * jnp.sum(mask * jnp.square(jax.lax.stop_gradient(Dm + delta) - Dm)) / npair''',
'''        if g_item is None:
            Gx, Gy = (gamma * (1 - d))[:, None], (gamma * (1 - d))[None, :]
            k_, nrm_ = k, 1.0 - gamma
        else:                                                           # 2026-11-09 n-step: per-item discount Gam_i
            Gx, Gy = g_item[:, None], g_item[None, :]
            k_, nrm_ = kn, 1.0 - gn
        vx, vy = 1 - Gx, 1 - Gy
        W1 = k_ * Gx * Gy; W2 = Gx * (vy - k_ * Gy); W3 = Gy * (vx - k_ * Gx); W4 = -vy - Gy * (vx - k_ * Gx)
        Dsg = jax.lax.stop_gradient(Dm)
        xi = vy * r[:, None] - vx * r[None, :] + W1 * M + W2 * Amx + W3 * Bmn + W4 * Dsg
        delta = jax.lax.stop_gradient(xi) / nrm_
        mask = 1.0 - jnp.eye(m)
        npair = m * (m - 1)
        if pmat is None:
            loss = 0.5 * jnp.sum(mask * jnp.square(jax.lax.stop_gradient(Dm + delta) - Dm)) / npair''')
rep('''            return duel_all_pairs(p, tp, obs, act, nxt, r_all, d_all,
                                  pmat=pw if (pw is not None and pw.ndim == 2) else None)''',
'''            g_item = tr.info.get("nstep_G", None) if (n_step > 1 and isinstance(tr.info, dict)) else None
            if n_step > 1:
                assert g_item is not None and al_mode == "none" and cw_closed == "none", "n_step: qcw/duel rho path only"
            return duel_all_pairs(p, tp, obs, act, nxt, r_all, d_all,
                                  pmat=pw if (pw is not None and pw.ndim == 2) else None, g_item=g_item)''')
# pass n_step to the compute_fn construction (second make_pair_loss call) -- inspect its kwargs
i=f.index('    _, compute_fn = make_pair_loss(net, float(config.system.gamma), str(config.system.term),')
j=f.index(')\n', f.index('al_alpha=float(config.system.get("al_alpha", 0.0))', i))
f=f[:j]+',\n                                   n_step=int(config.system.get("n_step", 1))'+f[j:]
# learner: n-step tail after the grouped batch
rep('''                transitions, ix, iy, pw, st, idx_ = same_traj_batch(buffer_state, sk1, m, st_group,
                                                                    int(config.arch.num_envs), return_idx=True)''',
'''                transitions, ix, iy, pw, st, idx_ = same_traj_batch(buffer_state, sk1, m, st_group,
                                                                    int(config.arch.num_envs), return_idx=True)
                n_step_ = int(config.system.get("n_step", 1))
                if n_step_ > 1:                                                 # 2026-11-09 n-step tails (same_traj.nstep_tail)
                    assert str(config.system.head) in ("duel", "qcw"), "n_step: qcw/duel heads only"
                    R_n, G_n, last_n, nu_n = nstep_tail(buffer_state, idx_, int(config.arch.num_envs), n_step_,
                                                        float(config.system.gamma), float(config.system.max_abs_reward))
                    exp_ = jax.tree_util.tree_map(lambda x: x[0], buffer_state.experience)
                    nxt_n = jax.tree_util.tree_map(lambda x: x[last_n], exp_.next_obs)
                    info_ = transitions.info if isinstance(transitions.info, dict) else {}
                    transitions = transitions._replace(reward=R_n, next_obs=nxt_n,
                                                       done=(G_n == 0.0).astype(transitions.done.dtype),
                                                       info={**info_, "nstep_G": G_n})
                    st = {**st, "nstep_used": jnp.mean(nu_n), "nstep_term_frac": jnp.mean((G_n == 0.0).astype(jnp.float32))}''')
open(D+'ff_rpair.py','w').write(f); print('patched')
