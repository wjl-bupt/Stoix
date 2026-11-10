"""2026-11-09 patch: closed-form CW-rho with separable E1 / E2 (system.cw_closed = none|A|B|C|D), all-pairs qcw/duel path.

Exact per-sample form of the implemented CW-rho gradient (theory doc 同轨迹配对的算法层面作用_理论进展_20261109.md, incl. the
duel_cur_target term): with w = pmat + pmat^T, v = 1 - g(1-d), T = d, N = 1 - d, u = r + g(1-d) Q-(s',a*) - Q(s,a),
g_cur = Q-(s,a) - Q(s,a) (0 without cur_target):
  e_i = om_i (u_i - v_i kap_i) + g T_i g_cur_i nu_i - g N_i om_i lam_i,     loss = -sum_i sg(e_i / (1-g)) Q(s_i,a_i)
  om_i = sum_j w_ij v_j (E2: trajectory-dependent sample weight),  nu_i = sum_j w_ij N_j
  kap_i = sum_j wk_ij u_j / sum_j wk_ij v_j,  lam_i = sum_j wk_ij T_j g_cur_j / sum_j wk_ij v_j   (E1: partners of the level)
Arms: A  om/nu realised, wk = w                 (== existing duel_all_pairs gradient)
      B  om/nu -> batch means (E2 off), wk = w
      C  om/nu realised, wk = w * cross-group     (E1 off: level only from independent partners)
      D  both off
Backup ff_rpair.py.bak_20261109_before_cwclosed.  Idempotent."""
import shutil
FR = "/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py"
s = open(FR).read()
if "cw_closed" in s:
    print("already patched"); raise SystemExit
shutil.copy(FR, FR + ".bak_20261109_before_cwclosed")

old = """                   al_mode: str = "none", al_alpha: float = 0.0):
    k = (1.0 - gamma) / gamma
"""
new = """                   al_mode: str = "none", al_alpha: float = 0.0, cw_closed: str = "none", st_group: int = 8):
    k = (1.0 - gamma) / gamma
    assert cw_closed in ("none", "A", "B", "C", "D")

    def closed_cw(A_on, A_n, A_sg, r, d, pmat):
        \"\"\"2026-11-09 closed-form CW-rho with separable E1/E2 (see patch_cw_closed.py docstring).\"\"\"
        m = r.shape[0]
        Qs = jax.lax.stop_gradient(A_on)
        Gd = gamma * (1.0 - d); v = 1.0 - Gd; T = d; N = 1.0 - d
        u = jax.lax.stop_gradient(r + Gd * A_n - Qs)
        gc = jax.lax.stop_gradient(A_sg - Qs) if cur_target else jnp.zeros_like(Qs)
        w = jax.lax.stop_gradient(pmat + pmat.T)
        grp = jnp.arange(m) // st_group
        cross = (grp[:, None] != grp[None, :]).astype(w.dtype)
        wk = w * cross if cw_closed in ("C", "D") else w
        om_r = w @ v; nu_r = w @ N
        if cw_closed in ("B", "D"):
            om = jnp.full_like(om_r, jnp.mean(om_r)); nu = jnp.full_like(nu_r, jnp.mean(nu_r))
        else:
            om, nu = om_r, nu_r
        den = wk @ v
        kap = (wk @ u) / den; lam = (wk @ (T * gc)) / den
        e = (om * (u - v * kap) + gamma * T * gc * nu - gamma * N * om * lam) / (1.0 - gamma)
        loss = -jnp.sum(jax.lax.stop_gradient(e) * A_on)
        wr = om_r / jnp.mean(om_r)
        info = {"q_loss": 0.5 * jnp.mean(jnp.square(e)), "td_std": jnp.std(u), "term_frac": jnp.mean(d),
                "e_abs": jnp.mean(jnp.abs(e)), "kap_mean": jnp.mean(kap), "kap_std": jnp.std(kap),
                "om_rel_max": jnp.max(wr), "om_rel_std": jnp.std(wr), "q_tilde_mean": jnp.mean(Qs)}
        return loss, info
"""
assert old in s; s = s.replace(old, new)

old2 = """        Dm = wmat(c_on, c_on) + A_on[:, None] - A_on[None, :]             # D((s_i,a_i),(s_j,a_j))
"""
new2 = """        if cw_closed != "none":                                         # 2026-11-09 closed-form CW-rho (E1/E2 switches)
            assert pmat is not None and net.head == "qcw", "cw_closed: qcw head with rho weight matrix only"
            return closed_cw(A_on, A_n, A_sg, r, d, pmat)
        Dm = wmat(c_on, c_on) + A_on[:, None] - A_on[None, :]             # D((s_i,a_i),(s_j,a_j))
"""
assert s.count(old2) == 1; s = s.replace(old2, new2)

old3 = """    _, compute_fn = make_pair_loss(net, float(config.system.gamma), str(config.system.term),
                                   float(config.system.cont_coef), float(config.system.max_abs_reward),
                                   all_pairs=bool(config.system.get("all_pairs", False)),
                                   cur_target=bool(config.system.get("duel_cur_target", False)),
"""
new3 = """    _, compute_fn = make_pair_loss(net, float(config.system.gamma), str(config.system.term),
                                   float(config.system.cont_coef), float(config.system.max_abs_reward),
                                   all_pairs=bool(config.system.get("all_pairs", False)),
                                   cur_target=bool(config.system.get("duel_cur_target", False)),
                                   cw_closed=str(config.system.get("cw_closed", "none")),
                                   st_group=int(config.system.get("st_group", 8)),
"""
assert s.count(old3) == 1; s = s.replace(old3, new3)
open(FR, "w").write(s)
print("patched")
