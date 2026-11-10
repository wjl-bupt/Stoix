# 2026-11-10 E1 causal control (user): for every TERMINAL item, its same-group (same-trajectory) pair weights are moved
# to the item in the same slot of a DIFFERENT, randomly mapped group (another trajectory), symmetrically.  Pair structure,
# weights, partner count and noise level of kappa_i are kept; only the link "terminal anchor <- own episode" is cut.
import shutil
D='/home/junweiluo/Stoix/stoix/systems/q_learning/'
for f in ['same_traj.py','ff_rpair.py']: shutil.copy(D+f, D+f+'.bak_20261110_before_tshuf')
s=open(D+'same_traj.py').read(); assert 'rho_weight_matrix_tshuffle' not in s
s+='''

def rho_weight_matrix_tshuffle(key, idx, G: int, rho: float, done):
    """2026-11-10 E1 control: rho_weight_matrix, then every same-group pair (i, k) with i terminal is moved to
    (i, sigma(k)), sigma(k) = slot of k in group h(group k), h = random fixed-point-free group map; symmetric, total
    weight 1 kept (a moved pair landing on the same buffer item as i is dropped).  Returns (W, moved_total)."""
    W = rho_weight_matrix(idx, G, rho)
    m = idx.shape[0]; n = m // G
    grp = jnp.arange(m) // G; slot = jnp.arange(m) % G
    perm = jax.random.permutation(key, n)
    pos = jnp.zeros(n, jnp.int32).at[perm].set(jnp.arange(n))
    h = perm[(pos + 1) % n]                                          # cyclic shift on a random order: h(g) != g
    sigma = h[grp] * G + slot
    d = done.astype(jnp.float32)
    same = (grp[:, None] == grp[None, :]).astype(jnp.float32)
    Wsame = W * same
    Erow = Wsame * d[:, None]                                        # pairs (i terminal, k same group)
    Esym = jnp.maximum(Erow, Erow.T)                                 # both orders of every pair with a terminal end
    moved = jnp.zeros((m, m), W.dtype).at[jnp.arange(m)[:, None], sigma[None, :]].add(Erow)
    moved = moved * (idx[:, None] != idx[None, :])
    W2 = W - Esym + moved + moved.T
    return W2, Erow.sum()
'''
open(D+'same_traj.py','w').write(s)
f=open(D+'ff_rpair.py').read()
o='rho_weight_matrix_sparse, nstep_tail'
assert f.count(o)==1; f=f.replace(o,o+', rho_weight_matrix_tshuffle')
o2='''                    elif int(config.system.get("rho_cross_k", 0)) > 0:'''
assert f.count(o2)==1
f=f.replace(o2,'''                    elif bool(config.system.get("rho_term_shuffle", False)):    # 2026-11-10 E1 causal control
                        pw, moved_ = rho_weight_matrix_tshuffle(sk2, idx_, st_group, pair_rho, transitions.done)
                        st = {**st, "tshuf_moved": moved_}
'''+o2)
open(D+'ff_rpair.py','w').write(f); print('patched')
