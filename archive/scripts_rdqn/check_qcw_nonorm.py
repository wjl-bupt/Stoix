"""2026-11-10 checks for cw_no_norm."""
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
T = SimpleNamespace(obs=jnp.asarray(O[oi]), next_obs=jnp.asarray(nxt), action=jnp.asarray(a), reward=jnp.asarray(rew), done=jnp.asarray(dn))
pw = rho_weight_matrix(jnp.arange(m), G, 0.5); ix = jnp.arange(m)
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
flat = lambda t: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(t)])
def grad(nn, gam=g):
    _, comp = make_pair_loss(net, gam, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True, cur_target=True, no_norm=nn)
    gr, info = jax.grad(lambda pp: comp(pp, tp, T, ix, ix, pw=pw), has_aux=True)(p)
    return flat(gr), gr, info
g0, _, i0 = grad(False); g1, gr1, i1 = grad(True)
rel = float(jnp.max(jnp.abs(g1 - (1 - g) * g0)) / jnp.max(jnp.abs(g0 * (1 - g))))
chk(rel < 1e-4, f"no_norm grad == (1-gamma) * original grad (max rel {rel:.1e})")
chk(abs(float(i1["td_std"]) / float(i0["td_std"]) - (1 - g)) < 1e-4, f"td_std scales by 1-gamma ({float(i1['td_std']):.4f} vs {float(i0['td_std']):.4f})")
bs = float(jnp.abs(gr1["params"]["du_adv"]["bias"].sum()) / jnp.max(jnp.abs(g1)))
chk(bs < 1e-5, f"gauge-neutral (|sum bias grad|/max|grad| {bs:.1e})")
ga, _, ia = grad(False, 0.5)
chk(bool(jnp.all(jnp.isfinite(ga))) and float(jnp.linalg.norm(ga - g0)) > 0, f"gamma=0.5 runs, finite grad (td_std {float(ia['td_std']):.3f})")
print("per-param |grad| median orig %.2e  no_norm %.2e  (Adam eps 1e-5)" % (float(jnp.median(jnp.abs(g0))), float(jnp.median(jnp.abs(g1)))))
print(f"{ok}/{n} passed")
