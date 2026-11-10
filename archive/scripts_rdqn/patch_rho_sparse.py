# 2026-11-09 (user): G=8, cross-trajectory rho weight shared by only K = G-1 partners per item (not the whole batch).
import shutil, re
D='/home/junweiluo/Stoix/stoix/systems/q_learning/'
for f in ['same_traj.py','ff_rpair.py']:
    shutil.copy(D+f, D+f+'.bak_20261109_before_rhosparse')
s=open(D+'same_traj.py').read()
assert 'def rho_weight_matrix_sparse' not in s
s+='''

def rho_weight_matrix_sparse(key, idx, G: int, rho: float, K: int):
    """2026-11-09 sparse cross part (user): like rho_weight_matrix, but every item has exactly K cross-trajectory
    partners (K = G - 1 by default) instead of all m - G items of the other groups.  Construction (symmetric, so the
    kernel stays gauge-neutral): the n_grp groups are randomly permuted and cut into blocks of K + 1 groups; inside
    every group the slots are randomly permuted; slot c of each group of a block forms a clique of K + 1 items (one
    per group = K + 1 independent anchors' trajectories).  Cross pairs = distinct-item pairs inside a clique.
    Same-group part unchanged ((1 - rho) over all same-group distinct-item pairs); cross part sums to rho.
    Requires n_grp % (K + 1) == 0."""
    m = idx.shape[0]
    n_grp = m // G
    assert n_grp % (K + 1) == 0, "rho_cross_k: number of groups must be a multiple of K + 1"
    kg, ks = jax.random.split(key)
    gperm = jax.random.permutation(kg, n_grp)                       # group at block position
    blk_of_grp = jnp.zeros(n_grp, jnp.int32).at[gperm].set(jnp.arange(n_grp) // (K + 1))
    slot_rank = jax.vmap(lambda k: jax.random.permutation(k, G))(jax.random.split(ks, n_grp))   # [n_grp, G]
    grp = jnp.arange(m) // G
    col = slot_rank.reshape(-1)                                      # clique column of each slot
    cid = blk_of_grp[grp] * G + col                                  # clique id
    distinct = idx[:, None] != idx[None, :]
    same = (grp[:, None] == grp[None, :]) & distinct
    cross = (cid[:, None] == cid[None, :]) & (grp[:, None] != grp[None, :]) & distinct
    Ws = same.astype(jnp.float32); Wc = cross.astype(jnp.float32)
    return (1.0 - rho) * Ws / jnp.maximum(Ws.sum(), 1.0) + rho * Wc / jnp.maximum(Wc.sum(), 1.0)
'''
open(D+'same_traj.py','w').write(s)
f=open(D+'ff_rpair.py').read()
old='from stoix.systems.q_learning.same_traj import same_traj_batch, rho_pairs, rho_weight_matrix, rho_weight_matrix_td, balance_kernel'
assert f.count(old)==1
f=f.replace(old, old+', rho_weight_matrix_sparse')
old2='''                    else:
                        pw = rho_weight_matrix(idx_, st_group, pair_rho)
'''
assert f.count(old2)==1
new2='''                    elif int(config.system.get("rho_cross_k", 0)) > 0:          # 2026-11-09 sparse cross partners (K per item)
                        pw = rho_weight_matrix_sparse(sk2, idx_, st_group, pair_rho, int(config.system.rho_cross_k))
                        st = {**st, "cross_deg": jnp.mean(jnp.sum((pw > 0) & ((jnp.arange(m) // st_group)[:, None] != (jnp.arange(m) // st_group)[None, :]), 1).astype(jnp.float32))}
                    else:
                        pw = rho_weight_matrix(idx_, st_group, pair_rho)
'''
f=f.replace(old2,new2)
open(D+'ff_rpair.py','w').write(f)
print('patched')
