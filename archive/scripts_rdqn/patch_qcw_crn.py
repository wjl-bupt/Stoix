"""2026-11-09 patch: same-state different-action pairs (CRN counterfactual branch) for CW-Q-rho (head=qcw, all pairs, rho kernel).

system.crn=True already makes every env step also execute b ~ U(A \\ {a}) from the same pre-step state with shared random
numbers (ff_rdqn.make_step_fn, cf_coupling=crn) and stores (b, r^b, d^b, s'^b) in the buffer item.  New here:
  nodes = m real transitions + their m counterfactual twins (same s_i, action b_i)
  kernel P2 = [[P_rho, c I], [c I, 0]],  c = same_coef / (2m)   (symmetric -> gauge neutral; cf nodes paired only with their twin)
  loss = existing exact duel_all_pairs (cross-weighted xi, cur_target kept) on the 2m nodes with kernel P2.
  same_coef = 0  ->  identical to CW-rho.  For two non-terminal branches xi = (1-g)(u_a - u_b): a level-free action-difference residual.
Mass convention as the old CW-Pair cf_pairs: cross-state pairs total 1, same-state pairs total same_coef.
Also: the CW-Pair-only counterfactual path (use_cf) is only built for head free/anti now (it asserted on qcw).
Backup ff_rpair.py.bak_20261109_before_qcwcrn.  Idempotent."""
import shutil
FR = "/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py"
s = open(FR).read()
if "cf_same" in s:
    print("already patched"); raise SystemExit
shutil.copy(FR, FR + ".bak_20261109_before_qcwcrn")

old = 'al_mode: str = "none", al_alpha: float = 0.0, cw_closed: str = "none", st_group: int = 8):'
assert s.count(old) == 1
s = s.replace(old, 'al_mode: str = "none", al_alpha: float = 0.0, cw_closed: str = "none", st_group: int = 8,\n                   cf_same: bool = False):')

old = """        if net.head in ("duel", "duel_sa", "qcw"):                                          # CW-Duel: state features only, all pairs
            assert all_pairs, "head=duel is implemented for system.all_pairs=True only"
"""
new = """        if net.head in ("duel", "duel_sa", "qcw"):                                          # CW-Duel: state features only, all pairs
            assert all_pairs, "head=duel is implemented for system.all_pairs=True only"
            if cf_same:                                                 # 2026-11-09 same-state (real, CRN counterfactual) pairs
                assert net.head == "qcw" and pw is not None and pw.ndim == 2, "cf_same: qcw + rho weight matrix"
                return qcw_cf_pairs(p, tp, tr, r_all, d_all, pw)
"""
assert s.count(old) == 1
s = s.replace(old, new)

old = """    def ap(p, method, *args):
        return net.apply(p, *args, method=method)
"""
new = """    def ap(p, method, *args):
        return net.apply(p, *args, method=method)

    def qcw_cf_pairs(p, tp, tr, r_real, d_real, pm):
        \"\"\"2026-11-09: CW-Q-rho on 2m nodes (real + CRN counterfactual twins), kernel [[pm, cI], [cI, 0]].\"\"\"
        m = r_real.shape[0]
        r_cf = jnp.clip(tr.cf_reward, -max_abs_reward, max_abs_reward).astype(jnp.float32)
        d_cf = tr.cf_done.astype(jnp.float32)
        cat = lambda a, b: jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y]), a, b)  # noqa: E731
        obs2, nxt2 = cat(tr.obs, tr.obs), cat(tr.next_obs, tr.cf_next_obs)
        act2 = jnp.concatenate([tr.action, tr.cf_action.astype(tr.action.dtype)])
        r2, d2 = jnp.concatenate([r_real, r_cf]), jnp.concatenate([d_real, d_cf])
        cI = jnp.eye(m, dtype=pm.dtype) * (same_coef / (2.0 * m))
        P2 = jnp.block([[pm, cI], [cI, jnp.zeros((m, m), pm.dtype)]])
        loss, info = duel_all_pairs(p, tp, obs2, act2, nxt2, r2, d2, pmat=P2)
        q_on = jax.lax.stop_gradient(ap(p, PairNet.duel_comp_obs, tr.obs)[3])
        gap = q_on[jnp.arange(m), tr.action] - q_on[jnp.arange(m), tr.cf_action.astype(tr.action.dtype)]
        same_out = jnp.all(jnp.equal(tr.next_obs, tr.cf_next_obs).reshape(m, -1), -1) & (r_real == r_cf) & (d_real == d_cf)
        info = {**info, "cf_term_frac": jnp.mean(d_cf), "same_dd_frac": jnp.mean(jnp.abs(d_real - d_cf)),
                "same_outcome_frac": jnp.mean(same_out.astype(jnp.float32)), "same_gap_abs": jnp.mean(jnp.abs(gap))}
        return loss, info
"""
assert s.count(old) == 1
s = s.replace(old, new)

old = "                                use_cf=bool(config.system.crn), same_coef=float(config.system.get(\"same_coef\", 1.0)),"
assert s.count(old) == 1
s = s.replace(old, "                                use_cf=bool(config.system.crn) and str(config.system.head) in (\"free\", \"anti\"),\n"
                   "                                same_coef=float(config.system.get(\"same_coef\", 1.0)),")

old = """                                   cw_closed=str(config.system.get("cw_closed", "none")),
                                   st_group=int(config.system.get("st_group", 8)),
"""
assert s.count(old) == 1
s = s.replace(old, old + """                                   cf_same=bool(config.system.crn) and str(config.system.head) == "qcw",
                                   same_coef=float(config.system.get("same_coef", 1.0)),
""")
open(FR, "w").write(s)
print("patched")
