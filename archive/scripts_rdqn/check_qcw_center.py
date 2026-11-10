import sys; sys.path.insert(0,'/home/junweiluo/Stoix')
import jax, jax.numpy as jnp, numpy as np
from stoix.systems.q_learning.ff_rpair import PairNet
ok=n=0
def chk(c,m):
    global ok,n; n+=1; ok+=bool(c); print('OK  ' if c else 'FAIL', m)
A=6; obs=jax.random.uniform(jax.random.PRNGKey(0),(64,10,10,4))
n0=PairNet(num_actions=A,head='qcw',rank=0); n1=PairNet(num_actions=A,head='qcw',rank=0,qcw_center='softmax')
p=n0.init(jax.random.PRNGKey(1),obs)
p=jax.tree_util.tree_map(lambda x: x*3.0, p)           # larger outputs so softmax is non-uniform
Q=n0.apply(p,obs,method=PairNet.duel_comp_obs)[3]; Qt=n1.apply(p,obs,method=PairNet.duel_comp_obs)[3]
chk(np.allclose(Qt, Q - (jax.nn.softmax(Q,-1)*Q).sum(-1,keepdims=True), atol=1e-5),'Qt = Q - E_softmax Q')
chk(np.abs((jax.nn.softmax(Q,-1)*Qt).sum(-1)).max()<1e-5,'E_{softmax(Q)} Qt = 0 per state')
chk((jnp.argmax(Q,-1)==jnp.argmax(Qt,-1)).all(),'argmax unchanged')
chk(np.allclose(n0.apply(p,obs,method=PairNet.scores), Q*A/(A-1)),'none = original')
chk(float(jnp.std((jax.nn.softmax(Q,-1)).max(-1)))>0,f'softmax non-trivial (max p mean {float(jax.nn.softmax(Q,-1).max(-1).mean()):.3f})')
g=jax.grad(lambda p_: jnp.sum(n1.apply(p_,obs,method=PairNet.duel_comp_obs)[3][:,0]))(p)
chk(all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(g)),'grad finite')
print(f'{ok}/{n}')
# --- softmax_sg ---
n2=PairNet(num_actions=A,head='qcw',rank=0,qcw_center='softmax_sg')
Qs=n2.apply(p,obs,method=PairNet.duel_comp_obs)[3]
chk(np.allclose(Qs,Qt,atol=1e-6),'softmax_sg forward == softmax forward')
f=lambda nn_: (lambda p_: jnp.sum(nn_.apply(p_,obs,method=PairNet.duel_comp_obs)[3]*jnp.arange(A)))
gq=jax.grad(f(n0))(p); gs=jax.grad(f(n2))(p); g1=jax.grad(f(n1))(p)
# manual: d/dQ of sum c*Qt with sg weights = c - p*sum(c)  -> chain through Q
P_=jax.nn.softmax(Q,-1); c=jnp.broadcast_to(jnp.arange(A,dtype=jnp.float32),Q.shape)
cm=c - P_*c.sum(-1,keepdims=True)
gman=jax.grad(lambda p_: jnp.sum(cm*n0.apply(p_,obs,method=PairNet.duel_comp_obs)[3]))(p)
fl=lambda t: np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(t)])
chk(np.allclose(fl(gs),fl(gman),atol=1e-4),'softmax_sg gradient == manual stop-grad gradient')
chk(not np.allclose(fl(gs),fl(g1),atol=1e-4),'softmax_sg gradient differs from full-gradient version')
print(f'{ok}/{n}')
