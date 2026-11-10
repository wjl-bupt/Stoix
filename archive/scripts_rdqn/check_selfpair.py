"""2026-11-09: what do self-pairs (pairs of the same buffer item, incl. the diagonal) do in the qcw CW-rho loss?
K_self: rho kernel that INCLUDES self-pairs (same-group/cross-group normalisation over all pairs incl. same item).
Check 1: gradient(K_self) == gradient(K_self with self entries zeroed)  -> self pairs contribute exactly 0.
Check 2: the remaining distinct-pair weights are a rescaled rho kernel: report effective rho and total mass."""
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
rng = np.random.default_rng(2); m, G = 512, 8
# realistic repeats: items drawn from a pool so that some buffer items appear twice (same item -> identical transition)
pool = rng.integers(0, 3000, m); pool[10] = pool[11]; pool[40] = pool[300]
oi, ni = rng.integers(0, len(O), 3000), rng.integers(0, len(O), 3000)
a_, r_, d_ = rng.integers(0, 3, 3000), (rng.random(3000) < .1).astype(np.float32), rng.random(3000) < .05
tr = SimpleNamespace(obs=jnp.asarray(O[oi[pool]]), next_obs=jnp.asarray(O[ni[pool]]), action=jnp.asarray(a_[pool]),
                     reward=jnp.asarray(r_[pool]), done=jnp.asarray(d_[pool]))
idx = jnp.asarray(pool); grp = jnp.arange(m) // G
same = (grp[:, None] == grp[None, :]).astype(jnp.float32); cross = 1.0 - same
Kself = 0.5 * same / same.sum() + 0.5 * cross / cross.sum()          # self pairs included in the normalisation
selfmask = (idx[:, None] == idx[None, :]).astype(jnp.float32)
Kzero = Kself * (1 - selfmask)
Korig = rho_weight_matrix(idx, G, 0.5)
_, comp = make_pair_loss(net, 0.99, "cross", float(S["cont_coef"]), float(S["max_abs_reward"]), all_pairs=True, cur_target=True)
gr = lambda K: jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(
    jax.grad(lambda pp: comp(pp, tp, tr, jnp.arange(m), jnp.arange(m), pw=K)[0])(p))])
g_self, g_zero, g_orig = gr(Kself), gr(Kzero), gr(Korig)
print("check 1: max|g(self incl.) - g(self zeroed)| / max|g| =", float(jnp.max(jnp.abs(g_self - g_zero)) / jnp.max(jnp.abs(g_self))))
ms = float((Kself * selfmask).sum()); s_d = float((Kzero * same).sum()); c_d = float((Kzero * cross).sum())
print(f"check 2: mass on self pairs {ms:.4f} (zero gradient); distinct same-group mass {s_d:.4f}, cross mass {c_d:.4f}; "
      f"effective rho = {c_d / (s_d + c_d):.4f}; total useful mass {s_d + c_d:.4f}")
Kr = rho_weight_matrix(idx, G, c_d / (s_d + c_d))
cos = float(g_self @ gr(Kr) / jnp.linalg.norm(g_self) / jnp.linalg.norm(gr(Kr)))
print(f"check 3: g(self incl.) vs g(rho kernel with rho_eff, no self): cosine {cos:.6f}, norm ratio {float(jnp.linalg.norm(g_self) / jnp.linalg.norm(gr(Kr))):.4f}")
print("vs original rho=0.5 kernel: cosine", float(g_self @ g_orig / jnp.linalg.norm(g_self) / jnp.linalg.norm(g_orig)))
