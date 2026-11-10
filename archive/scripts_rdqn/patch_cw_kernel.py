"""2026-11-09 patch: gauge-neutral E1/E2 factorial via symmetric pair kernels (system.cw_kernel = none|B2|C2).

Lesson of the cw_closed run: the CW residual u_i - v_i kappa_i is invariant to Q -> Q + c, so the update must satisfy
sum_i e_i = 0 (no push along the level direction; otherwise the output bias drifts without restoring force: arms B/D
with constant omega went to -1e9).  e_i = sum_j w_ij xi_ij with SYMMETRIC w and antisymmetric xi guarantees it, so
the factorial is built from symmetric kernels and the existing exact pairwise loss (duel_all_pairs, cur_target kept):
  B2 (E1 on,  E2 off): w' = diag(s) w_rho diag(s), s chosen so sum_j w'_ij v_j = const for all i
  C2 (E1 off, E2 on ): w' = diag(s) w_cross diag(s), s chosen so sum_j w'_ij v_j = omega_i^A = sum_j w_rho,ij v_j
  D  (both off)      : pair_rho = 1.0 (cross-group pairs only, uniform) -- existing code, no patch needed
s by symmetric Sinkhorn iterations s <- s * sqrt(target / (s * (w (s*v)))) ; result rescaled to total weight 1.
same_traj.py: + balance_kernel(W, v, target, iters);  ff_rpair.py: apply in the rho branch.  Backups *.bak_20261109_before_cwkernel."""
import shutil
B = "/home/junweiluo/Stoix/stoix/systems/q_learning/"
ST, FR = B + "same_traj.py", B + "ff_rpair.py"
s, f = open(ST).read(), open(FR).read()
if "def balance_kernel" in s:
    print("already patched"); raise SystemExit
for x in (ST, FR):
    shutil.copy(x, x + ".bak_20261109_before_cwkernel")
s = s.rstrip("\n") + '''


def balance_kernel(W, v, target, iters: int = 40):
    """2026-11-09 symmetric Sinkhorn balancing: returns symmetric w' = diag(s) (W + W^T) diag(s) / Z with
    sum_j w'_ij v_j proportional to target_i (exactly, up to iteration error), total weight 1, zero pattern of W kept."""
    Ws = W + W.T
    def body(_, sc):
        rows = sc * (Ws @ (sc * v))
        return sc * jnp.sqrt(target / jnp.maximum(rows, 1e-30))
    sc = jax.lax.fori_loop(0, iters, body, jnp.ones_like(v))
    w2 = sc[:, None] * Ws * sc[None, :]
    return jax.lax.stop_gradient(w2 / jnp.sum(w2))
'''
if "import jax\n" not in s and "import jax\r\n" not in s:
    s = s.replace("import jax.numpy as jnp", "import jax\nimport jax.numpy as jnp", 1)
open(ST, "w").write(s)

old_imp = "rho_weight_matrix, rho_weight_matrix_td\n"
assert old_imp in f
f = f.replace(old_imp, "rho_weight_matrix, rho_weight_matrix_td, balance_kernel\n")
old = """                    else:
                        pw = rho_weight_matrix(idx_, st_group, pair_rho)
                    ix = iy = jnp.arange(m)
"""
new = """                    else:
                        pw = rho_weight_matrix(idx_, st_group, pair_rho)
                    cw_kernel = str(config.system.get("cw_kernel", "none"))     # 2026-11-09 gauge-neutral E1/E2 factorial
                    if cw_kernel != "none":
                        v_ = 1.0 - float(config.system.gamma) * (1.0 - transitions.done.astype(jnp.float32))
                        wA = pw + pw.T
                        omA = wA @ v_
                        if cw_kernel == "B2":
                            pw = balance_kernel(pw, v_, jnp.full_like(omA, jnp.mean(omA)))
                        elif cw_kernel == "C2":
                            g_ = jnp.arange(m) // st_group
                            pc = rho_weight_matrix(idx_, st_group, 1.0) * (g_[:, None] != g_[None, :])
                            pw = balance_kernel(pc, v_, omA)
                        else:
                            raise ValueError(cw_kernel)
                        wN = pw + pw.T; omN = wN @ v_
                        st = {**st, "omA_rel_std": jnp.std(omA / jnp.mean(omA)),
                              "om_rel_std": jnp.std(omN / jnp.mean(omN)),
                              "om_match": jnp.max(jnp.abs(omN / jnp.mean(omN) - (omA / jnp.mean(omA) if cw_kernel == "C2" else 1.0)))}
                    ix = iy = jnp.arange(m)
"""
assert f.count(old) == 1
f = f.replace(old, new)
open(FR, "w").write(f)
print("patched")
