# 2026-11-09 (user): qcw head output Qt(s,a) = Q(s,a) - sum_b softmax(Q(s,.))_b Q(s,b)  (softmax-weighted zero mean per state,
# temperature 1, gradient flows through the weights); used everywhere the qcw Q is used (pair differences, bootstrap, policy).
import shutil
F='/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py'
shutil.copy(F, F+'.bak_20261109_before_qcwcenter')
s=open(F).read()
o1='''    vmax: float = 20.0      # rc51: support [-vmax, vmax]
'''
assert s.count(o1)==1
s=s.replace(o1,o1+'''    qcw_center: str = "none"  # 2026-11-09 qcw: "softmax" -> Q - E_{softmax(Q)}[Q] per state
''')
o2='''            Ac = self.du_adv(h)
            z = jnp.zeros(h.shape[:-1] + (0,), h.dtype)'''
assert s.count(o2)==1
s=s.replace(o2,'''            Ac = self.du_adv(h)
            if self.qcw_center == "softmax":
                Ac = Ac - jnp.sum(jax.nn.softmax(Ac, -1) * Ac, -1, keepdims=True)
            elif self.qcw_center != "none":
                raise ValueError(self.qcw_center)
            z = jnp.zeros(h.shape[:-1] + (0,), h.dtype)''')
o3='''                  additive=bool(config.system.get("all_pairs", False)) and head in ("free", "anti"))
    train_eps'''
assert s.count(o3)==1
s=s.replace(o3,'''                  additive=bool(config.system.get("all_pairs", False)) and head in ("free", "anti"),
                  qcw_center=str(config.system.get("qcw_center", "none")))
    train_eps''')
open(F,'w').write(s); print('patched')
