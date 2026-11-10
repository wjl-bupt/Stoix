# 2026-11-09 (user): qcw_center="softmax_sg": Q - sum_b sg(softmax(Q))_b Q(s,b)  (weights without gradient)
import shutil
F='/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py'
shutil.copy(F, F+'.bak_20261109_before_qcwcentersg')
s=open(F).read()
o='''            elif self.qcw_center != "none":
                raise ValueError(self.qcw_center)'''
assert s.count(o)==1
s=s.replace(o,'''            elif self.qcw_center == "softmax_sg":     # 2026-11-09: same centring, no gradient through the weights
                Ac = Ac - jnp.sum(jax.lax.stop_gradient(jax.nn.softmax(Ac, -1)) * Ac, -1, keepdims=True)
            elif self.qcw_center != "none":
                raise ValueError(self.qcw_center)''')
open(F,'w').write(s); print('patched')
