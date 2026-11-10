# 2026-11-09 (user): advantage learning (Bellemare et al. 2016, AL, beta=al_alpha) for the all-pairs heads (qcw / duel).
# Per item: r_i <- r_i - alpha * G-(s_i,a_i),  G- = max_b Q-(s_i,b) - Q-(s_i,a_i) from the TARGET net (stop-grad).
# Identical to the existing CW-Pair form xi' = xi - alpha (v_y P(x) - v_x P(y)) and to DQN+AL (y - alpha P);
# G- is invariant to Q -> Q + c(s), so the CW residual stays gauge-neutral.
import shutil
F='/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rpair.py'
shutil.copy(F, F+'.bak_20261109_before_qcwal')
s=open(F).read()
o1='''    if al_mode != "none":
        assert al_mode in ("al", "pal") and net.head in ("free", "anti") and term == "cross" and not all_pairs \\
            and not use_cf, "AL/PAL: CW-Pair free/anti, term=cross, K partners / same_traj only"'''
assert s.count(o1)==1, 'o1'
s=s.replace(o1,'''    if al_mode != "none":
        assert (al_mode in ("al", "pal") and net.head in ("free", "anti") and term == "cross" and not all_pairs
                and not use_cf) or (al_mode == "al" and net.head in ("qcw", "duel") and all_pairs and term == "cross"
                and not use_cf), "AL/PAL: CW-Pair free/anti (K partners / same_traj) or AL for all-pairs qcw/duel"''')
o2='''        if cw_closed != "none":                                         # 2026-11-09 closed-form CW-rho (E1/E2 switches)'''
assert s.count(o2)==1, 'o2'
s=s.replace(o2,'''        al_pen = jnp.zeros_like(r)
        if al_mode == "al":                                             # 2026-11-09 AL: r - alpha * gap of target net at s
            c_al = c_sg if cur_target else ap(tp, PairNet.duel_comp_obs, obs)
            al_pen = jax.lax.stop_gradient(jnp.max(c_al[3], -1) - c_al[3][jnp.arange(m), act])
            r = r - al_alpha * al_pen
'''+o2)
o3='''                "adv_abs": jnp.mean(jnp.abs(Ac))}
        if return_delta:'''
k=s.index(o2); j=s.index(o3,k); assert s.count(o3)>=1
s=s[:j]+s[j:].replace(o3,'''                "adv_abs": jnp.mean(jnp.abs(Ac)), "al_pen": jnp.mean(al_pen)}
        if return_delta:''',1)
open(F,'w').write(s); print('patched')
