"""2026-11-09 patch: terminal-decoupled rho (system.rho_term_decouple=True) for the all-pairs heads (qcw/duel).
same_traj.py : + rho_weight_matrix_td(idx, G, rho, done)
ff_rpair.py  : use it when config.system.rho_term_decouple is set; log td_removed (weight moved per batch).
Backups *.bak_20261109_before_rhotd.  Idempotent."""
import shutil
B = "/home/junweiluo/Stoix/stoix/systems/q_learning/"
ST, FR = B + "same_traj.py", B + "ff_rpair.py"
for f in (ST, FR):
    shutil.copy(f, f + ".bak_20261109_before_rhotd")

FUNC = '''

def rho_weight_matrix_td(idx, G: int, rho: float, done):
    """2026-11-09 terminal-decoupled rho (diagnosis of the Breakout survival deficit, mechanisms M1/M2):
    same as rho_weight_matrix, but every SAME-group pair in which either item is terminal (done) gets weight 0; the
    weight removed from row i is added to row i's cross-group pairs in proportion to their weights, so every row sum
    (hence each item's total weight as the x side) and the total (1) are unchanged.  A terminal item thus forms its
    level estimate kappa_i only from independent (cross-trajectory) partners: removes the v_late * eta_early bias of
    same-trajectory pairs and the noisy same-trajectory kappa (denominator ~ (1 - gamma)).
    Returns (W [m, m], removed_total)."""
    W = rho_weight_matrix(idx, G, rho)
    m = idx.shape[0]
    grp = jnp.arange(m) // G
    same = grp[:, None] == grp[None, :]
    cross = (~same) & (idx[:, None] != idx[None, :])
    d = done.astype(bool)
    rm = same & (d[:, None] | d[None, :])
    Wrm = jnp.where(rm, W, 0.0)
    keep = W - Wrm
    removed_row = Wrm.sum(1)
    cross_row = jnp.where(cross, keep, 0.0).sum(1)
    scale = 1.0 + removed_row / jnp.maximum(cross_row, 1e-30)
    W2 = jnp.where(cross, keep * scale[:, None], keep)
    return W2, Wrm.sum()
'''
s = open(ST).read()
if "def rho_weight_matrix_td" not in s:
    s = s.rstrip("\n") + "\n" + FUNC
    open(ST, "w").write(s)

f = open(FR).read()
old_imp = "from stoix.systems.q_learning.same_traj import same_traj_batch, rho_pairs, rho_weight_matrix\n"
new_imp = "from stoix.systems.q_learning.same_traj import same_traj_batch, rho_pairs, rho_weight_matrix, rho_weight_matrix_td\n"
assert old_imp in f or new_imp in f
f = f.replace(old_imp, new_imp)
old = """                if str(config.system.head) in ("duel", "duel_sa", "qcw"):      # all-pairs head: weight matrix instead of a pair list
                    pw = rho_weight_matrix(idx_, st_group, pair_rho)
                    ix = iy = jnp.arange(m)
"""
new = """                if str(config.system.head) in ("duel", "duel_sa", "qcw"):      # all-pairs head: weight matrix instead of a pair list
                    if bool(config.system.get("rho_term_decouple", False)):    # 2026-11-09 terminal-decoupled rho
                        pw, td_removed = rho_weight_matrix_td(idx_, st_group, pair_rho, transitions.done)
                        st = {**st, "td_removed": td_removed}
                    else:
                        pw = rho_weight_matrix(idx_, st_group, pair_rho)
                    ix = iy = jnp.arange(m)
"""
if "rho_term_decouple" not in f:
    assert old in f
    f = f.replace(old, new)
    open(FR, "w").write(f)
print("patched")
