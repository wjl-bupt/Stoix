"""2026-11-09 checks for AL in the all-pairs qcw head (system.al_mode=al)."""
import json, pickle, sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, "/home/junweiluo/Stoix")
import jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.same_traj import rho_weight_matrix
R = "/home/junweiluo/stoix_runs/minatar_g8ck/"
S = json.loads(open(R + "breakout_qcw_rho_g8ck_s0.jsonl").readline())["config"]["system"]
net = PairNet(num_actions=3, head="qcw", hidden=128, emb_dim=16, rank=0, n_atoms=51, vmax=20.0, additive=False)
ld = lambda f: jax.tree_util.tree_map(lambda x: x[0], pickle.load(open(R + "ckpt/" + f, "rb")))
p, tp = ld("breakout_qcw_rho_g8ck_s0_t4998400.pkl"), ld("breakout_qcw_rho_g8ck_s0_t3998720.pkl")
O = np.asarray(pickle.load(open("/home/junweiluo/stoix_runs/breakout_miss/bank.pkl", "rb")).observation)
rng = np.random.default_rng(3); m, G, g, al = 512, 8, 0.99, 0.9
oi = rng.integers(0, len(O), m); a = rng.integers(0, 3, m)
rew = (rng.random(m) < .1).astype(np.float32)
mk_tr = lambda rw: SimpleNamespace(obs=jnp.asarray(O[oi]), next_obs=jnp.asarray(O[rng2.integers(0, len(O), m)]), action=jnp.asarray(a),
                                   reward=jnp.asarray(rw), done=jnp.asarray(dn))
rng2 = np.random.default_rng(5); dn = rng.random(m) < .05
nxt = O[rng2.integers(0, len(O), m)]
def T(rw): return SimpleNamespace(obs=jnp.asarray(O[oi]), next_obs=jnp.asarray(nxt), action=jnp.asarray(a), reward=jnp.asarray(rw), done=jnp.asarray(dn))
pw = rho_weight_matrix(jnp.arange(m), G, 0.5); ix = jnp.arange(m)
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
def mk(mode, alpha):
    _, comp = make_pair_loss(net, g, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True,
                             cur_target=True, al_mode=mode, al_alpha=alpha)
    return comp
flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
def grad(comp, tr):
    return jax.grad(lambda pp: comp(pp, tp, tr, ix, ix, pw=pw), has_aux=True)(p)
g_base, _ = grad(mk("none", 0.0), T(rew))
g0, i0 = grad(mk("al", 0.0), T(rew))
chk(float(jnp.max(jnp.abs(flat(g0) - flat(g_base)))) == 0.0, "al_alpha=0 identical to original")
g1, i1 = grad(mk("al", al), T(rew))
Qt = net.apply(tp, jnp.asarray(O[oi]), method=PairNet.duel_comp_obs)[3]
Gap = np.asarray(Qt.max(-1) - Qt[jnp.arange(m), jnp.asarray(a)])
chk((Gap >= 0).all() and abs(float(i1["al_pen"]) - Gap.mean()) < 1e-5, f"al_pen = target gap >= 0 (mean {Gap.mean():.4f})")
g_r, _ = grad(mk("none", 0.0), T(rew - al * Gap))
d = float(jnp.max(jnp.abs(flat(g1) - flat(g_r))) / jnp.max(jnp.abs(flat(g_r))))
chk(d < 1e-5, f"AL == original loss with rewards r - alpha*gap (max rel diff {d:.1e})")
dn_ = float(jnp.linalg.norm(flat(g1) - flat(g_base)) / jnp.linalg.norm(flat(g_base)))
chk(dn_ > 1e-3, f"AL changes the gradient (rel norm {dn_:.3f})")
bs = float(jnp.abs(g1["params"]["du_adv"]["bias"].sum()) / jnp.max(jnp.abs(flat(g1))))
chk(bs < 1e-5, f"gauge-neutral with AL (|sum bias grad|/max|grad| {bs:.1e})")
# gap invariance: shifting the target net's output bias leaves the AL gradient unchanged
tp2 = jax.tree_util.tree_map(lambda x: x, tp); tp2["params"]["du_adv"]["bias"] = tp["params"]["du_adv"]["bias"] + 3.0
_, comp = make_pair_loss(net, g, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True, cur_target=True, al_mode="al", al_alpha=al)
g2, _ = jax.grad(lambda pp: comp(pp, tp2, T(rew), ix, ix, pw=pw), has_aux=True)(p)
d2 = float(jnp.max(jnp.abs(flat(g2) - flat(g1))) / jnp.max(jnp.abs(flat(g1))))
chk(d2 < 1e-4, f"target-net constant shift leaves AL gradient unchanged (max rel diff {d2:.1e})")
print(f"{ok}/{n} passed")
