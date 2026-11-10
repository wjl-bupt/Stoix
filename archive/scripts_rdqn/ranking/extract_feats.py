"""2026-11-09 linear probe, step 1: penultimate features phi(s) (MinAtarTorso output, 128-d, post-ReLU) of every
Breakout checkpoint on the shared bank states.  Writes one npz: feats[name] [N, 128], qnet[name] [N, A], obs [N, 400].
usage: extract_feats.py <bank.pkl> <out.npz> <ckpt.pkl>...  (CPU)"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, "/home/junweiluo/Stoix")
import jax, jax.numpy as jnp  # noqa: E402
from stoix.systems.q_learning.ff_rdqn import MinAtarTorso  # noqa: E402
bank = pickle.load(open(sys.argv[1], "rb"))
obs = jnp.asarray(np.asarray(bank.observation)).astype(jnp.float32)
torso = MinAtarTorso()
out = {"obs": np.asarray(obs).reshape(len(obs), -1)}
for c in sys.argv[3:]:
    p = jax.tree_util.tree_map(lambda x: x[0], pickle.load(open(c, "rb")))
    tp = p["q"]["params"] if "q" in p else p["params"]
    head = tp["action_head"]["Dense_0"] if "action_head" in tp else tp["du_adv"]
    f = torso.apply({"params": tp["torso"]}, obs)
    q = f @ head["kernel"] + head["bias"]
    n = os.path.basename(c)[:-4]
    out[f"feat__{n}"] = np.asarray(f); out[f"q__{n}"] = np.asarray(q)
np.savez_compressed(sys.argv[2], **out)
print("ok", len(sys.argv) - 3, obs.shape)
