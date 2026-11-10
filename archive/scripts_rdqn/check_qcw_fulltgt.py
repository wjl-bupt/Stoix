"""2026-11-10 checks for qcw_full_target."""
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
rng = np.random.default_rng(3); m, G, g = 512, 8, 0.99
oi = rng.integers(0, len(O), m); a = rng.integers(0, 3, m); rew = (rng.random(m) < .1).astype(np.float32)
nxt = O[rng.integers(0, len(O), m)]; dn = rng.random(m) < .05
T = lambda d: SimpleNamespace(obs=jnp.asarray(O[oi]), next_obs=jnp.asarray(nxt), action=jnp.asarray(a), reward=jnp.asarray(rew), done=jnp.asarray(d))
pw = rho_weight_matrix(jnp.arange(m), G, 0.5); ix = jnp.arange(m)
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
def grad(ft, tparams, d):
    _, comp = make_pair_loss(net, g, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True, cur_target=True, full_target=ft)
    return flat(jax.grad(lambda pp: comp(pp, tparams, T(d), ix, ix, pw=pw), has_aux=True)(p)[0])
rel = lambda x, y: float(jnp.max(jnp.abs(x - y)) / jnp.max(jnp.abs(y)))
g0 = grad(False, tp, dn); g1 = grad(True, tp, dn)
chk(rel(grad(False, p, dn), grad(True, p, dn)) < 1e-5, "target == online => identical gradient")
nd = np.zeros(m, bool)
chk(rel(grad(True, tp, nd), grad(False, tp, nd)) < 1e-4, "no terminals => identical gradient even with target != online")
r = float(jnp.linalg.norm(g1 - g0) / jnp.linalg.norm(g0))
chk(r > 1e-3, f"with terminals and target != online the gradient changes (rel norm {r:.3f})")
# theory: delta_cur - delta_full = g/(1-g) (D^- - D) on pairs with >=1 terminal  => grad difference equals grad of that extra term
Qo = net.apply(p, jnp.asarray(O[oi]), method=PairNet.duel_comp_obs)[3][jnp.arange(m), jnp.asarray(a)]
Qt = net.apply(tp, jnp.asarray(O[oi]), method=PairNet.duel_comp_obs)[3][jnp.arange(m), jnp.asarray(a)]
dd = jnp.asarray(dn, jnp.float32); oneplus = 1 - (1 - dd[:, None]) * (1 - dd[None, :])
E = jax.lax.stop_gradient(g / (1 - g) * oneplus * ((Qt[:, None] - Qt[None, :]) - (Qo[:, None] - Qo[None, :])))
def lin(pp):
    Q = net.apply(pp, jnp.asarray(O[oi]), method=PairNet.duel_comp_obs)[3][jnp.arange(m), jnp.asarray(a)]
    return -jnp.sum(pw * E * (Q[:, None] - Q[None, :]))
gt = flat(jax.grad(lin)(p))
chk(rel(g0 - g1, gt) < 1e-3, f"grad difference == theory g/(1-g)(D^- - D) on terminal pairs (rel {rel(g0 - g1, gt):.1e})")
_, comp = make_pair_loss(net, g, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True, cur_target=True, full_target=True)
gg = jax.grad(lambda pp: comp(pp, tp, T(dn), ix, ix, pw=pw), has_aux=True)(p)[0]
bs = float(jnp.abs(gg["params"]["du_adv"]["bias"].sum()) / jnp.max(jnp.abs(g1)))
chk(bs < 1e-5, f"gauge-neutral (|sum bias grad|/max|grad| {bs:.1e})")
print(f"{ok}/{n} passed")
