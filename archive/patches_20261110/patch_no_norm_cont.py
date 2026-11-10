"""2026-11-10 (user: nonorm is the default of CW-Q-rho): optional no_norm for the continuous code.
 - stoix/systems/ddpg/cw_continuous.py : +system.no_norm=True  -> T = sg[D + xi] instead of sg[D + xi/(1-gamma)]
 - cwq/mujoco_gym/mujoco_offpolicy.py  : --no_norm 1            (same)
Python defaults stay OFF (= the code that produced all earlier runs); the cwq/ launchers pass the flag explicitly."""
import shutil, sys
S = "/home/junweiluo/Stoix/"
F = S + "stoix/systems/ddpg/cw_continuous.py"
shutil.copy(F, F + ".bak_20261110_before_nonorm")
s = open(F).read()
a = 'def make_critic_fns(net, head: str, gamma: float, n_critics: int, min_mode: str = "raw"):\n    """Returns critic_loss(qp, tqp, tr, a_ref, a_n, a_n_ref, pairs) -> (loss, info) and actor_value(...)."""\n'
assert s.count(a) == 1
s = s.replace(a, a.replace('min_mode: str = "raw"):', 'min_mode: str = "raw", no_norm: bool = False):')
              + "    nrm = 1.0 if no_norm else (1.0 - gamma)          # 2026-11-10: no_norm drops the 1/(1-gamma) of xi\n")
assert s.count("xi / (1.0 - gamma)") == 3; s = s.replace("xi / (1.0 - gamma)", "xi / nrm")
b = 'str(config.system.get("min_mode", "raw")))'
assert s.count(b) == 1; s = s.replace(b, 'str(config.system.get("min_mode", "raw")),\n                                               bool(config.system.get("no_norm", False)))')
open(F, "w").write(s)
F = S + "cwq/mujoco_gym/mujoco_offpolicy.py"
shutil.copy(F, F + ".bak_20261110_before_nonorm")
s = open(F).read()
a = 'def critic_loss_fn(qnet, qp, tqp, b, a_n, gamma, head, W, min_mode="raw"):'
assert s.count(a) == 1; s = s.replace(a, a[:-2] + ', no_norm=0):')
a = "T = jax.lax.stop_gradient(D + xi / (1.0 - gamma))"
assert s.count(a) == 1; s = s.replace(a, "T = jax.lax.stop_gradient(D + xi / (1.0 if no_norm else 1.0 - gamma))   # 2026-11-10 no_norm")
a = 'ap.add_argument("--min_mode", default="raw", choices=["raw", "centered", "none"])'
assert s.count(a) == 1; s = s.replace(a, a + '\n    ap.add_argument("--no_norm", type=int, default=0)        # 2026-11-10: 1 = CW residual without 1/(1-gamma)')
a = "A.head, W, A.min_mode)"
assert s.count(a) >= 1; n = s.count(a); s = s.replace(a, "A.head, W, A.min_mode, A.no_norm)")
open(F, "w").write(s); print("patched; mujoco call sites:", n)
