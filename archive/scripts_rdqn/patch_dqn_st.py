import shutil,sys
f="/home/junweiluo/Stoix/stoix/systems/q_learning/ff_rdqn.py"
s=open(f).read()
old='        assert rdqn and not (use_traj or crn or rdqn_a or c51 or reward_centering), "same_traj: plain RDQN only"'
new=('        # 2026-11-08: same_traj also allowed for plain DQN (rdqn=False): only the batch construction changes\n'
     '        # (grouped same-trajectory sampler); the DQN loss ignores the pairs.\n'
     '        assert not (use_traj or crn or rdqn_a or c51 or reward_centering), "same_traj: plain DQN/RDQN only"\n'
     '        assert rdqn or pair_rho is None, "pair_rho needs rdqn"')
assert s.count(old)==1, "anchor not found"
shutil.copy(f, f+".bak_20261108_before_dqn_st")
open(f,"w").write(s.replace(old,new)); print("patched")
