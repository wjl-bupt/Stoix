"""g-score probe (2026-11-03), adapted from Zhou et al. 2022 (camall3n/actgen change_q.py / gscore.py).

Question: when training pushes ONE training unit up, how much of that push becomes an action-specific change at the
unit's own state (what the greedy policy needs), and how much leaks to other actions / other states?

usage: CUDA_VISIBLE_DEVICES=k .venv/bin/python gscore.py <state_dir> <dqn|cw> <out_json> [--check]

Training unit u and its trained output f_theta(u):
  dqn / rdqn : u = replay item x0 = (s0, a0),          f = q_theta(s0, a0)          (RDQN: the scalar c is not in theta)
  cw (CW-Pair): u = (x0, y), y = an independent replay item (the training pairing),  f = D_theta(x0, y)
Greedy read-out v_theta(s, b):  dqn/rdqn  q_theta(s, b);   cw  Borda score(s, b) = PairNet.scores (the policy's read-out).
Push: a gradient step that raises f by delta, theta' = theta + eta * delta * P grad f(u), where
  P = I (plain SGD, as in the original g-score) or P = Adam preconditioner diag(1/(sqrt(nu_hat)+eps)) from the captured
  optimizer state (what training actually applies, momentum ignored).  To first order in eta every read-out moves by
  dv(s,b) = eta*delta*<grad v(s,b), P grad f(u)>, and f itself by df = eta*delta*<grad f, P grad f>.  All quantities
  below are ratios, so eta*delta cancels; they are computed exactly with one VJP and one JVP (first-order / NTK limit).
  --check compares them with a real finite SGD step (eta small) to confirm the linearisation.
Metrics per probe (aggregated by median and by ratio of sums over probes):
  kappa   = dv(s0,a0) / df                         read-out transfer (1 by construction for dqn/rdqn)
  g_same  = mean_{b!=a0} dv(s0,b) / dv(s0,a0)      same-state coupling: 1 = the push is a pure level shift of s0
  tau     = [dv(s0,a0) - mean_{b!=a0} dv(s0,b)] / df    ACTION-SPECIFIC TRANSFER: fraction of the trained change that
                                                         becomes the action gap of a0 at s0  (= kappa (1 - g_same))
  lev_X   = mean_{s in X} mean_b dv(s,b) / dv(s0,a0)   level leakage to state set X
  rel_X   = mean_{s in X} mean_{b<b'} |dv(s,b)-dv(s,b')| / |dv(s0,a0) - mean_{b!=a0} dv(s0,b)|
                                                         ranking leakage: action-difference change at X per unit of
                                                         action-difference change at s0
  X = 'other' (512 independent replay items) and 'traj' (up to 8 other items of x0's own trajectory, |dt| recorded).
"""
import json, os, pickle, sys, time
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, pgx
from jax.flatten_util import ravel_pytree
jax.config.update("jax_default_matmul_precision", "highest")   # no TF32: ratios of small first-order changes
sys.path.insert(0, "/home/junweiluo/Stoix")
from stoix.systems.q_learning.ff_rpair import PairNet
from stoix.systems.q_learning.ff_rdqn import MinAtarTorso
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import DiscreteQNetworkHead

ROOT, ALGO, OUTF = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
CHECK = "--check" in sys.argv
NPROBE, NOTHER, NTRAJ, CH = (16, 64, 8, 4) if CHECK else (256, 512, 8, 8)
AEPS, B2 = 1e-5, 0.999
t0 = time.time()
cfg = json.load(open(ROOT / "capture_config.json"))
task = cfg["env"]["scenario"]["task_name"]
with open(ROOT / "full_state.pkl", "rb") as f:
    state = jax.tree_util.tree_map(jnp.asarray, pickle.load(f))
env = pgx.make(task if task.startswith("minatar") else "minatar-" + task); A = env.num_actions
E = int(cfg["arch"]["num_envs"])
bs = state.buffer_state
exp = jax.tree_util.tree_map(lambda x: x[0], bs.experience)
L = exp.done.shape[0]; p = int(bs.current_index); n_valid = L if bool(bs.is_full) else p
OBS = exp.obs.astype(jnp.float32); ACT = exp.action

if ALGO == "cw":
    net = PairNet(num_actions=A, head="free")
    th = state.params.online
    readout = lambda t, o: net.apply(t, o, method=PairNet.scores)
    def f_unit(t, o_x, a_x, o_y, a_y):
        ex = net.apply(t, net.apply(t, o_x[None], method=PairNet.phi), a_x[None], method=PairNet.sa)
        ey = net.apply(t, net.apply(t, o_y[None], method=PairNet.phi), a_y[None], method=PairNet.sa)
        return net.apply(t, ex, ey, method=PairNet.D)[0]
else:
    qnet = FeedForwardActor(torso=MinAtarTorso(), action_head=DiscreteQNetworkHead(action_dim=A, epsilon=0.0))
    th = state.params.online["q"]
    readout = lambda t, o: qnet.apply(t, o).preferences
    def f_unit(t, o_x, a_x, o_y, a_y):
        return qnet.apply(t, o_x[None]).preferences[0, a_x]

F64 = bool(jax.config.read("jax_enable_x64"))            # --check is run with JAX_ENABLE_X64=1 on CPU
if F64:
    th = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, th)
    OBS = OBS.astype(jnp.float64)
# Adam preconditioner from the captured optimizer state (chain(clip, adam)): ScaleByAdamState(count, mu, nu)
adam = state.opt_states["q"][1][0]
cnt = int(adam.count)
nu_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - B2 ** cnt), adam.nu)
Pre = jax.tree_util.tree_map(lambda v: 1.0 / (jnp.sqrt(v) + AEPS), nu_hat)
flat_th, unravel = ravel_pytree(th)
print("STATE", ALGO, task, "adam count", cnt, "replay", n_valid, "params", flat_th.shape[0], flush=True)

# ---- same-trajectory layout (same arithmetic as same_traj.py): item at offset o back from p: env (-o) mod E ----
J = (L + E - 1) // E + 1
o_grid = E * (J - np.arange(J))[None, :] - np.arange(E)[:, None]
valid = (o_grid >= 1) & (o_grid <= n_valid)
pos_grid = np.mod(p - o_grid, L)
done_np = np.asarray(exp.done).astype(bool)
dg = np.where(valid, done_np[pos_grid], True).astype(np.int64)
ep_id = np.cumsum(dg, 1) - dg

rng = np.random.default_rng(12345)
o_probe = rng.integers(1, n_valid + 1, NPROBE)
e_p = np.mod(-o_probe, E); tau_p = J - (o_probe + e_p) // E
pos_probe = np.mod(p - o_probe, L)
traj_pos = np.full((NPROBE, NTRAJ), -1); traj_dt = np.zeros((NPROBE, NTRAJ))
for i in range(NPROBE):
    row = ep_id[e_p[i]]; ids = np.where(row == row[tau_p[i]])[0]; ids = ids[ids != tau_p[i]]
    if len(ids):
        pick = rng.choice(ids, size=min(NTRAJ, len(ids)), replace=False)
        traj_pos[i, :len(pick)] = np.mod(p - (E * (J - pick) - e_p[i]), L); traj_dt[i, :len(pick)] = np.abs(pick - tau_p[i])
traj_mask = traj_pos >= 0
traj_pos = np.where(traj_mask, traj_pos, pos_probe[:, None])
other_pos = np.mod(p - rng.integers(1, n_valid + 1, NOTHER), L)
partner_pos = np.mod(p - rng.integers(1, n_valid + 1, NPROBE), L)      # CW pairing partner y of each probe

# sanity: the probe's next_obs equals the obs of the next item of its trajectory (when not done)
chk = []
for i in range(min(NPROBE, 64)):
    row = ep_id[e_p[i]]
    if tau_p[i] + 1 < J and row[tau_p[i] + 1] == row[tau_p[i]] and not done_np[pos_probe[i]]:
        nxt = np.mod(p - (E * (J - (tau_p[i] + 1)) - e_p[i]), L)
        chk.append(bool(np.array_equal(np.asarray(exp.next_obs[pos_probe[i]]), np.asarray(exp.obs[nxt]))))
print("layout check", np.mean(chk) if chk else None, len(chk), flush=True)

O_other = OBS[other_pos]

def one(o0, a0, oy, ay, o_traj, precond):
    g = jax.grad(f_unit)(th, o0, a0, oy, ay)
    d = jax.tree_util.tree_map(lambda gg, pp: gg * pp, g, Pre) if precond else g
    df = sum(jnp.vdot(a, b) for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(d)))
    allo = jnp.concatenate([o0[None], o_traj, O_other], 0)
    _, dv = jax.jvp(lambda t: readout(t, allo), (th,), (d,))
    return df, dv            # dv: [1 + NTRAJ + NOTHER, A]
batched = {pc: jax.jit(jax.vmap(lambda *a, pc=pc: one(*a, pc))) for pc in (False, True)}

def metrics(df, dv, a0, tmask):
    s0 = dv[0]; v0 = s0[a0]; oth = np.delete(s0, a0).mean()
    gap0 = v0 - oth
    def lev(block): return block.mean(-1)
    def rel(block):
        diffs = np.abs(block[:, :, None] - block[:, None, :]); iu = np.triu_indices(A, 1)
        return diffs[:, iu[0], iu[1]].mean(-1)
    tr = dv[1:1 + NTRAJ][tmask]; ot = dv[1 + NTRAJ:]
    return dict(df=df, dv0=v0, gap0=gap0, oth0=oth,
                lev_other=lev(ot).mean(), rel_other=rel(ot).mean(),
                lev_traj=lev(tr).mean() if len(tr) else np.nan, rel_traj=rel(tr).mean() if len(tr) else np.nan)

res = {}
for pc in (False, True):
    rows = []
    for s in range(0, NPROBE, CH):
        sl = slice(s, s + CH)
        df, dv = batched[pc](OBS[pos_probe[sl]], ACT[pos_probe[sl]], OBS[partner_pos[sl]], ACT[partner_pos[sl]],
                             OBS[traj_pos[sl]])
        df, dv = np.asarray(df, np.float64), np.asarray(dv, np.float64)
        for k in range(df.shape[0]):
            i = s + k
            rows.append(metrics(df[k], dv[k], int(ACT[pos_probe[i]]), traj_mask[i]))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    agg = {}
    agg["kappa_med"] = float(np.median(R["dv0"] / R["df"]))
    agg["kappa_sum"] = float(R["dv0"].sum() / R["df"].sum())
    agg["g_same_med"] = float(np.median(R["oth0"] / R["dv0"]))
    agg["g_same_sum"] = float(R["oth0"].sum() / R["dv0"].sum())
    agg["tau_med"] = float(np.median(R["gap0"] / R["df"]))
    agg["tau_sum"] = float(R["gap0"].sum() / R["df"].sum())
    for X in ("other", "traj"):
        ok = np.isfinite(R[f"lev_{X}"])
        agg[f"lev_{X}_med"] = float(np.median(R[f"lev_{X}"][ok] / R["dv0"][ok]))
        agg[f"lev_{X}_sum"] = float(R[f"lev_{X}"][ok].sum() / R["dv0"][ok].sum())
        agg[f"rel_{X}_med"] = float(np.median(R[f"rel_{X}"][ok] / np.abs(R["gap0"][ok])))
        agg[f"rel_{X}_sum"] = float(R[f"rel_{X}"][ok].sum() / np.abs(R["gap0"][ok]).sum())
    agg["frac_dv0_pos"] = float(np.mean(R["dv0"] > 0)); agg["frac_gap0_pos"] = float(np.mean(R["gap0"] > 0))
    res["adam" if pc else "sgd"] = agg
    res[("adam" if pc else "sgd") + "_raw"] = {k: v.tolist() for k, v in R.items()}
res["traj_dt_mean"] = float(traj_dt[traj_mask].mean()); res["traj_n_mean"] = float(traj_mask.sum(1).mean())
res["layout_check"] = float(np.mean(chk)) if chk else None
res["meta"] = dict(algo=ALGO, task=task, dir=str(ROOT), adam_count=cnt, n_probe=NPROBE, n_other=NOTHER)

if CHECK:   # finite-step check of the linearisation (plain SGD step, eta small)
    errs = []
    for i in range(4):
        o0, a0, oy, ay = OBS[pos_probe[i]], ACT[pos_probe[i]], OBS[partner_pos[i]], ACT[partner_pos[i]]
        allo = jnp.concatenate([o0[None], OBS[traj_pos[i]], O_other], 0)
        g = jax.grad(f_unit)(th, o0, a0, oy, ay)
        df_lin, dv_lin = batched[False](o0[None], a0[None], oy[None], ay[None], OBS[traj_pos[i]][None])
        for eta in (1e-3, 1e-4):
            th2 = jax.tree_util.tree_map(lambda t, gg: t + eta * gg, th, g)
            dv_fin = (readout(th2, allo) - readout(th, allo)) / eta
            df_fin = (f_unit(th2, o0, a0, oy, ay) - f_unit(th, o0, a0, oy, ay)) / eta
            errs.append(dict(i=i, eta=eta, rel_err_dv=float(jnp.linalg.norm(dv_fin - dv_lin[0]) / jnp.linalg.norm(dv_lin[0])),
                             rel_err_df=float(abs(df_fin - df_lin[0]) / abs(df_lin[0]))))
    res["linearisation_check"] = errs
    for e in errs: print("LIN", e, flush=True)
res["wall"] = time.time() - t0
OUTF.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUTF, "w"))
print(json.dumps({k: v for k, v in res.items() if not k.endswith("_raw")}, indent=1), flush=True)
