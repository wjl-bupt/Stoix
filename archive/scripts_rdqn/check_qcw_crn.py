"""2026-11-09 checks for qcw same-state CRN pairs (cf_same)."""
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
oi = rng.integers(0, len(O), m); a = rng.integers(0, 3, m); b = (a + 1 + rng.integers(0, 2, m)) % 3
tr = SimpleNamespace(obs=jnp.asarray(O[oi]), next_obs=jnp.asarray(O[rng.integers(0, len(O), m)]), action=jnp.asarray(a),
                     reward=jnp.asarray((rng.random(m) < .1).astype(np.float32)), done=jnp.asarray(rng.random(m) < .05),
                     cf_action=jnp.asarray(b), cf_next_obs=jnp.asarray(O[rng.integers(0, len(O), m)]),
                     cf_reward=jnp.asarray((rng.random(m) < .1).astype(np.float32)), cf_done=jnp.asarray(rng.random(m) < .05))
pw = rho_weight_matrix(jnp.arange(m), G, 0.5); ix = jnp.arange(m)
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
def mk(cf, sc):
    _, comp = make_pair_loss(net, g, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True,
                             cur_target=True, cf_same=cf, same_coef=sc)
    return comp
flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
def grad(comp):
    gr, info = jax.grad(lambda pp: comp(pp, tp, tr, ix, ix, pw=pw), has_aux=True)(p); return gr, info
g_base, _ = grad(mk(False, 1.0)); g0, _ = grad(mk(True, 0.0)); g1, info1 = grad(mk(True, 1.0))
d0 = float(jnp.max(jnp.abs(flat(g0) - flat(g_base))) / jnp.max(jnp.abs(flat(g_base))))
chk(d0 < 1e-5, f"same_coef=0 reproduces CW-rho gradient (max rel diff {d0:.1e})")
bs = float(jnp.abs(g1["params"]["du_adv"]["bias"].sum()) / jnp.max(jnp.abs(flat(g1))))
chk(bs < 1e-5, f"same_coef=1 gauge-neutral (|sum bias grad|/max|grad| {bs:.1e})")
dn = float(jnp.linalg.norm(flat(g1) - flat(g0)) / jnp.linalg.norm(flat(g0)))
chk(dn > 1e-3, f"same-state pairs change the gradient (rel norm {dn:.3f})")
# independent hand computation of the same-state part: loss_s = -sum_i c * sg(delta_ab) * (Q(s,a) - Q(s,b)) * 2 (both orders)
def q_on(pp, o, act): return pp and net.apply(pp, o, method=PairNet.duel_comp_obs)[3][jnp.arange(m), act]
def q_t(o, act): return net.apply(tp, o, method=PairNet.duel_comp_obs)[3][jnp.arange(m), act]
def vmax_t(o): q = net.apply(tp, o, method=PairNet.duel_comp_obs)[3]; return q[jnp.arange(m), jnp.argmax(q, -1)]
da, db = tr.done.astype(jnp.float32), tr.cf_done.astype(jnp.float32)
Ga, Gb = g * (1 - da), g * (1 - db); va, vb = 1 - Ga, 1 - Gb; k = (1 - g) / g
def loss_s(pp):
    Qa, Qb = q_on(pp, tr.obs, tr.action), q_on(pp, tr.obs, tr.cf_action)
    Qa_s, Qb_s = jax.lax.stop_gradient(Qa), jax.lax.stop_gradient(Qb)
    ua = tr.reward + Ga * vmax_t(tr.next_obs) - Qa_s; ub = tr.cf_reward + Gb * vmax_t(tr.cf_next_obs) - Qb_s
    ga, gb = q_t(tr.obs, tr.action) - Qa_s, q_t(tr.obs, tr.cf_action) - Qb_s           # cur_target terms
    W3 = Gb * (va - k * Ga); W2 = Ga * (vb - k * Gb)
    xi = vb * ua - va * ub + W3 * ga - W2 * gb
    delta = jax.lax.stop_gradient(xi / (1 - g)); D = Qa - Qb
    c = 1.0 / (2 * m)
    return 0.5 * c * 2 * jnp.sum(jnp.square(jax.lax.stop_gradient(D + delta) - D))     # (a,b) and (b,a) give the same term
g_s = jax.grad(loss_s)(p)
dd = float(jnp.max(jnp.abs(flat(g1) - flat(g0) - flat(g_s))) / jnp.max(jnp.abs(flat(g_s))))
chk(dd < 1e-4, f"same-state contribution == independent hand computation (max rel diff {dd:.1e})")
nn = (1 - da) * (1 - db)
chk(True, f"info: same_gap_abs {float(info1['same_gap_abs']):.3f}, same_dd_frac {float(info1['same_dd_frac']):.3f}")
print(f"{ok}/{n} passed")
