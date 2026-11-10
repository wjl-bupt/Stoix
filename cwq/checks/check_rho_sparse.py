import jax, jax.numpy as jnp, numpy as np, sys
sys.path.insert(0,'/home/junweiluo/Stoix')
from stoix.systems.q_learning.same_traj import rho_weight_matrix_sparse, rho_weight_matrix
ok=0; n=0
def chk(c,msg):
    global ok,n; n+=1; ok+=bool(c); print(('OK  ' if c else 'FAIL'),msg)
m,G,K,rho=512,8,7,0.5
for t in range(5):
    key=jax.random.PRNGKey(t)
    idx=jax.random.randint(jax.random.PRNGKey(100+t),(m,),0,100000)
    if t==4: idx=idx.at[9].set(idx[0]).at[100].set(idx[3])   # forced duplicate items
    W=np.array(rho_weight_matrix_sparse(key,idx,G,rho,K)); I=np.array(idx)
    grp=np.arange(m)//G; same=grp[:,None]==grp[None,:]
    chk(np.allclose(W,W.T,atol=0),f'[{t}] symmetric')
    chk(abs(W.sum()-1)<1e-5 and abs(W[same].sum()-(1-rho))<1e-5 and abs(W[~same].sum()-rho)<1e-5,f'[{t}] sums 1 / 1-rho / rho')
    deg=((W>0)&~same).sum(1)
    chk((deg<=K).all() and (deg==K).mean()>0.99, f'[{t}] cross degree K (min {deg.min()} max {deg.max()} frac==K {(deg==K).mean():.4f})')
    # each partner from a different group
    okg=all(len(set(grp[np.where((W[i]>0)&~same[i])[0]]))==deg[i] for i in range(m))
    chk(okg,f'[{t}] cross partners in distinct groups')
    chk((W[I[:,None]==I[None,:]]==0).all(),f'[{t}] no same-item pairs')
    cw=W[~same]; cw=cw[cw>0]; chk(np.allclose(cw,cw[0]),f'[{t}] uniform cross weight {cw[0]:.3e} (=rho/(m K)={rho/(m*K):.3e})')
    # gauge neutrality: antisymmetric xi -> sum of per-item gradient mass zero
    A=np.random.RandomState(t).randn(m,m); xi=A-A.T
    e=(W*xi).sum(1); chk(abs(e.sum())<1e-6,f'[{t}] gauge neutral sum e={e.sum():.1e}')
    # same-group part identical to dense version
    W0=np.array(rho_weight_matrix(idx,G,rho)); chk(np.allclose(W[same],W0[same]),f'[{t}] same-group part == dense')
# randomness: different keys give different cliques
W1=np.array(rho_weight_matrix_sparse(jax.random.PRNGKey(0),idx,G,rho,K)); W2=np.array(rho_weight_matrix_sparse(jax.random.PRNGKey(1),idx,G,rho,K))
chk(((W1>0)!=(W2>0)).any(),'different keys -> different partners')
# coverage: over many keys every cross pair appears roughly uniformly
cnt=np.zeros((m,m))
for t in range(300): cnt+=np.array(rho_weight_matrix_sparse(jax.random.PRNGKey(1000+t),jnp.arange(m),G,rho,K))>0
cc=cnt[~same]; exp=300*K/(m-G)
chk(abs(cc.mean()-exp)/exp<0.02,f'coverage uniform: mean {cc.mean():.3f} vs {exp:.3f}, std {cc.std():.3f}')
print(f'{ok}/{n}')
