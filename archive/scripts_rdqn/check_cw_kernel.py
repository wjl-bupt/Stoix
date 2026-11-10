"""2026-11-09 checks for the gauge-neutral kernels (balance_kernel; arms A / B2 / C2 / D=rho 1)."""
import json, pickle, sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, "/home/junweiluo/Stoix")
import jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.same_traj import rho_weight_matrix, balance_kernel
R = "/home/junweiluo/stoix_runs/minatar_g8ck/"
cfg = json.loads(open(R + "breakout_qcw_rho_g8ck_s0.jsonl").readline())["config"]; S = cfg["system"]
net = PairNet(num_actions=3, head="qcw", hidden=128, emb_dim=16, rank=0, n_atoms=51, vmax=20.0, additive=False)
ld = lambda f: jax.tree_util.tree_map(lambda x: x[0], pickle.load(open(R + "ckpt/" + f, "rb")))
p, tp = ld("breakout_qcw_rho_g8ck_s0_t4998400.pkl"), ld("breakout_qcw_rho_g8ck_s0_t3998720.pkl")
O = np.asarray(pickle.load(open("/home/junweiluo/stoix_runs/breakout_miss/bank.pkl", "rb")).observation)
rng = np.random.default_rng(1); m, G, g = 512, 8, 0.99
done = jnp.asarray(rng.random(m) < .05)
tr = SimpleNamespace(obs=jnp.asarray(O[rng.integers(0, len(O), m)]), next_obs=jnp.asarray(O[rng.integers(0, len(O), m)]),
                     action=jnp.asarray(rng.integers(0, 3, m)), reward=jnp.asarray((rng.random(m) < .1).astype(np.float32)), done=done)
idx = jnp.arange(m).at[3].set(2); ix = jnp.arange(m)
v = 1 - g * (1 - done.astype(jnp.float32)); grp = np.arange(m) // G
pw = rho_weight_matrix(idx, G, 0.5); omA = (pw + pw.T) @ v
pc = rho_weight_matrix(idx, G, 1.0) * (grp[:, None] != grp[None, :])
K = {"A": pw, "B2": balance_kernel(pw, v, jnp.full_like(omA, omA.mean())), "C2": balance_kernel(pc, v, omA), "D": rho_weight_matrix(idx, G, 1.0)}
ok = n = 0
def chk(c, msg):
    global ok, n
    n += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + msg)
for k, W in K.items():
    W = np.asarray(W); om = (W + W.T) @ np.asarray(v)
    chk(np.allclose(W, W.T, atol=1e-12) or k in ("A", "D"), f"{k}: symmetric (A/D symmetric by construction of w = W + W^T)")
    chk(abs(W.sum() - 1) < 1e-4, f"{k}: total weight 1 ({W.sum():.6f})")
    chk(W[2, 3] == 0 and W[3, 2] == 0 and np.all(np.diag(W) == 0), f"{k}: same-item pairs zero")
    if k == "B2":
        chk(np.max(np.abs(om / om.mean() - 1)) < 1e-3, f"B2: sample weights equalised (max rel dev {np.max(np.abs(om / om.mean() - 1)):.1e}; A had std {float(jnp.std(omA / omA.mean())):.3f})")
    if k == "C2":
        tgt = np.asarray(omA / omA.mean())
        chk(np.max(np.abs(om / om.mean() - tgt)) < 1e-3, f"C2: sample weights == A's (max abs dev {np.max(np.abs(om / om.mean() - tgt)):.1e})")
        chk(np.all(W[grp[:, None] == grp[None, :]] == 0), "C2: no same-group pairs (level from independent partners only)")
# gauge neutrality: sum over actions of the output-bias gradient == sum_i e_i  -> must vanish
def bias_sum(mode, pmat):
    _, comp = make_pair_loss(net, g, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True,
                             cur_target=True, cw_closed=mode, st_group=G)
    gr, _ = jax.grad(lambda pp: comp(pp, tp, tr, ix, ix, pw=pmat), has_aux=True)(p)
    b = gr["params"]["du_adv"]["bias"]; full = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(gr)])
    return float(jnp.abs(b.sum()) / jnp.max(jnp.abs(full)))
for k, W in K.items():
    r = bias_sum("none", W); chk(r < 1e-4, f"{k}: gauge-neutral, |sum of bias grad| / max|grad| = {r:.1e}")
r = bias_sum("B", K["A"]); chk(r > 1e-2, f"control: old closed-form arm B (constant omega) is NOT gauge-neutral ({r:.2e})")
print(f"{ok}/{n} passed")
