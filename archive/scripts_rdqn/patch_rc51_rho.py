# 2026-11-09 (user): R-C51 with the rho pairing (same_traj groups G=8, rho_pairs: all within-group ordered pairs weighted
# (1-rho), K=7 cross-group partners per slot weighted rho) -- same pair structure as CW-Pair-rho.  Only change to R-C51:
# the pair cross-entropy is the pw-weighted sum instead of the plain mean (pw=None keeps the old behaviour).
import shutil
F='/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py'
shutil.copy(F, F+'.bak_20261109_before_rc51rho')
s=open(F).read()
def rep(o,n_,cnt=1):
    global s
    assert s.count(o)==cnt, (o[:60], s.count(o)); s=s.replace(o,n_)
rep('''    def rc51_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all, ix, iy):''',
    '''    def rc51_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all, ix, iy, pw=None):''')
rep('''        ce = -jnp.mean(jnp.sum(tgt * jax.nn.log_softmax(logit_on, -1), -1))''',
    '''        ce_j = -jnp.sum(tgt * jax.nn.log_softmax(logit_on, -1), -1)
        ce = jnp.mean(ce_j) if pw is None else jnp.sum(pw * ce_j)     # 2026-11-09: rho-kernel pair weights (sum 1)''')
rep('''            return rc51_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all, ix, iy)''',
    '''            return rc51_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all, ix, iy, pw=pw)''')
rep('''            or (str(config.system.head) in ("duel", "duel_sa", "qcw") and config.system.get("pair_rho", None) is not None), \\
            "same_traj: free/anti + cross (or head=duel with pair_rho), plain replay only"''',
    '''            or (str(config.system.head) in ("duel", "duel_sa", "qcw") and config.system.get("pair_rho", None) is not None) \\
            or (str(config.system.head) == "rc51" and config.system.get("pair_rho", None) is not None and not weighted
                and not bool(config.system.get("all_pairs", False)) and not bool(config.system.crn)), \\
            "same_traj: free/anti + cross (or head=duel/qcw or rc51 with pair_rho), plain replay only"''')
open(F,'w').write(s); print('patched')
