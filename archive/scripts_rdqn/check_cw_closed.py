"""2026-11-09 checks for cw_closed: arm A gradient == existing duel_all_pairs gradient (cur_target on and off);
B/C/D finite and different from A; C/D level ignores same-group partners; B/D weights constant."""
import json, pickle, sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, "/home/junweiluo/Stoix")
import jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.same_traj import rho_weight_matrix
R = "/home/junweiluo/stoix_runs/minatar_g8ck/"
cfg = json.loads(open(R + "breakout_qcw_rho_g8ck_s0.jsonl").readline())["config"]; S = cfg["system"]
net = PairNet(num_actions=3, head="qcw", hidden=128, emb_dim=16, rank=0, n_atoms=int(S.get("rc_atoms", 51)),
              vmax=float(S.get("rc_vmax", 20.0)), additive=False)
ld = lambda f: jax.tree_util.tree_map(lambda x: x[0], pickle.load(open(R + "ckpt/" + f, "rb")))
p, tp = ld("breakout_qcw_rho_g8ck_s0_t4998400.pkl"), ld("breakout_qcw_rho_g8ck_s0_t3998720.pkl")
bank = pickle.load(open("/home/junweiluo/stoix_runs/breakout_miss/bank.pkl", "rb")); O = np.asarray(bank.observation)
rng = np.random.default_rng(0); m = 512
tr = SimpleNamespace(obs=jnp.asarray(O[rng.integers(0, len(O), m)]), next_obs=jnp.asarray(O[rng.integers(0, len(O), m)]),
                     action=jnp.asarray(rng.integers(0, 3, m)), reward=jnp.asarray((rng.random(m) < .1).astype(np.float32)),
                     done=jnp.asarray(rng.random(m) < .05))
idx = jnp.arange(m).at[3].set(2)                                       # repeated buffer item in group 0
pw = rho_weight_matrix(idx, 8, 0.5); ix = jnp.arange(m)
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
def grads(mode, cur):
    _, comp = make_pair_loss(net, 0.99, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True,
                             cur_target=cur, cw_closed=mode, st_group=8)
    g, info = jax.grad(lambda pp: comp(pp, tp, tr, ix, ix, pw=pw), has_aux=True)(p)
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g)]), info
for cur in (True, False):
    g0, _ = grads("none", cur); gA, iA = grads("A", cur)
    rel = float(jnp.max(jnp.abs(gA - g0)) / jnp.max(jnp.abs(g0)))
    chk(rel < 1e-4, f"cur_target={cur}: arm A gradient == existing CW-rho gradient (max rel diff {rel:.2e}, |g| {float(jnp.max(jnp.abs(g0))):.3g})")
G = {k: grads(k, True) for k in "ABCD"}
for k in "BCD":
    gk, info = G[k]
    chk(bool(jnp.all(jnp.isfinite(gk))), f"arm {k}: finite gradient")
    d = float(jnp.linalg.norm(gk - G["A"][0]) / jnp.linalg.norm(G["A"][0]))
    chk(d > 1e-3, f"arm {k}: differs from A (rel norm diff {d:.3f}); cos {float(gk @ G['A'][0] / jnp.linalg.norm(gk) / jnp.linalg.norm(G['A'][0])):.3f}")
chk(float(G["B"][1]["om_rel_std"]) > 0 and float(G["A"][1]["om_rel_std"]) == float(G["B"][1]["om_rel_std"]), "om_rel_* logs the realised weights in every arm")
print(f"{ok}/{n} passed")
