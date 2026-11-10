import sys; sys.path.insert(0,'/home/junweiluo/Stoix')
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.same_traj import rho_weight_matrix, rho_weight_matrix_tshuffle
ok=n=0
def chk(c,m):
    global ok,n; n+=1; ok+=bool(c); print('PASS' if c else 'FAIL', m)
m,G=512,8; rng=np.random.default_rng(0)
for t in range(4):
    idx=jnp.asarray(rng.integers(0,10**6,m)); done=np.zeros(m,bool)
    term_groups=rng.choice(m//G,10,replace=False)
    for g in term_groups: done[g*G+rng.integers(G)]=True          # <=1 terminal per trajectory group
    W0=np.array(rho_weight_matrix(idx,G,.5)); W,mv=rho_weight_matrix_tshuffle(jax.random.PRNGKey(t),idx,G,.5,jnp.asarray(done)); W=np.array(W)
    grp=np.arange(m)//G; same=grp[:,None]==grp[None,:]; d=done
    chk(np.allclose(W,W.T),f'[{t}] symmetric'); chk(abs(W.sum()-1)<1e-5,f'[{t}] total weight 1 ({W.sum():.6f})')
    chk((W[d][:, :][same[d]]==0).all(),f'[{t}] terminal rows have no same-group weight')
    term_pair=(d[:,None]|d[None,:])
    chk(np.allclose(W[~term_pair],W0[~term_pair]) or True, f'[{t}] (info) non-terminal pairs mostly unchanged')
    nt=~term_pair; diff=np.abs(W-W0)[nt]
    # non-terminal pairs change only where a moved terminal weight landed: those pairs involve a terminal end, so nt pairs must be identical
    chk(diff.max()<1e-6,f'[{t}] pairs without a terminal end unchanged (max diff {diff.max():.1e})')
    rs=W[d].sum(1)/W0[d].sum(1); chk(np.abs(rs-1).max()<0.2,f'[{t}] terminal row sums ~kept (ratio min {rs.min():.3f} max {rs.max():.3f}, mean {rs.mean():.4f})')
    i=np.where(d)[0][0]; tg=np.unique(grp[np.where(W[i]>W0[i]*1.01+1e-12)[0]])
    chk(len(tg)==1 and tg[0]!=grp[i],f'[{t}] moved weight of terminal row goes to exactly one other group')
    A=rng.normal(size=(m,m)); xi=A-A.T; chk(abs((W*xi).sum())<1e-9,f'[{t}] gauge-neutral (sum e = {(W*xi).sum():.1e})')
print(f'{ok}/{n}')
