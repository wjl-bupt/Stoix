"""Checks for ff_rpair.py (2026-10-31).  JAX_PLATFORMS=cpu python scripts_rdqn/check_rpair.py
T1  tabular mock: exact expectation of delta over both transitions' outcomes, at consistent D = Q*(x) - Q*(y):
    cross -> 0 for every pair (random / always / never terminating pairs);  unif with exact head -> 0.
T2  same with a shape-perturbed Q: expectations not all zero (the check can fail).
T3  unif with a wrong head: biased (nonzero at Q*); cross needs no head.
T4  pair-type formulas of cross (both continue / x ends / both end).
T5  PairNet: shapes, antisymmetry and D(x,x)=0 for head=anti, not for head=free; Borda read-out = argmax Q for
    a consistent table (via the mock).
"""
import sys, types
sys.path.insert(0, "/home/junweiluo/Stoix")
import numpy as np, jax, jax.numpy as jnp
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss
from stoix.systems.q_learning.dqn_types import Transition

ok = tot = 0
def chk(name, cond, extra=""):
    global ok, tot
    tot += 1; ok += bool(cond); print(("PASS " if cond else "FAIL ") + name + " " + extra)

class Mock:
    """tabular stand-in for PairNet: e = state*A + action, D(ex,ey) = T[ex] - T[ey]."""
    head = "mock"
    def __init__(self, A): self.A = A
    def apply(self, p, *args, method):
        n = method.__name__; T = p["T"]; A = self.A
        if n == "phi": return args[0]
        if n == "sa": return args[0] * A + args[1]
        if n == "D": return T[args[0]] - T[args[1]]
        if n == "all_e": return args[0][:, None] * A + jnp.arange(A)[None, :]
        if n == "scores_from_e":
            v = T[args[0]]; return (A * v - v.sum(-1, keepdims=True)) / (A - 1)
        if n == "cont_logit": return p["L"][args[0]]
        raise ValueError(n)

rng = np.random.default_rng(0)
S, A, g = 6, 3, 0.9
P = rng.dirichlet(np.ones(S) * .7, size=(S, A))
pt = rng.uniform(0, .6, size=(S, A)); pt[0, :] = 0.0; pt[1, 0] = 1.0           # never / always / random
r = rng.normal(size=(S, A))
Q = np.zeros((S, A))
for _ in range(3000): Q = r + g * (1 - pt) * (P @ Q.max(1))
X = S * A

def outcomes(x):
    s, a = divmod(x, A); out = []
    if pt[s, a] > 0: out.append((pt[s, a], 1, 0))
    for s2 in range(S):
        w = (1 - pt[s, a]) * P[s, a, s2]
        if w > 0: out.append((w, 0, s2))
    return out

# build all (pair, outcome-combo) rows
xs, ys = [], []
rows_x, rows_y, wts, pid = [], [], [], []
k = 0
for x in range(X):
    for y in range(X):
        if x == y: continue
        for (wx, dx, sx) in outcomes(x):
            for (wy, dy, sy) in outcomes(y):
                rows_x.append((x, dx, sx)); rows_y.append((y, dy, sy)); wts.append(wx * wy); pid.append(k)
        k += 1
npairs = k; N = len(wts)
def mk(rows):
    rows = np.array(rows)
    s, a = np.divmod(rows[:, 0], A)
    return dict(obs=s, action=a, reward=r[s, a], done=rows[:, 1].astype(bool), next_obs=rows[:, 2])
bx, by = mk(rows_x), mk(rows_y)
tr = Transition(obs=jnp.array(np.concatenate([bx["obs"], by["obs"]])),
                action=jnp.array(np.concatenate([bx["action"], by["action"]])),
                reward=jnp.array(np.concatenate([bx["reward"], by["reward"]]), jnp.float32),
                done=jnp.array(np.concatenate([bx["done"], by["done"]])),
                next_obs=jnp.array(np.concatenate([bx["next_obs"], by["next_obs"]])), info={})
perm = jnp.array(np.concatenate([np.arange(N) + N, np.arange(N)]))
wts = np.array(wts); pid = np.array(pid)
mock = Mock(A)

def expected_delta(term, Tq, logit):
    _, compute = make_pair_loss(mock, g, term, 0.33, 1e9, return_delta=True)
    p = {"T": jnp.array(Tq.reshape(-1), jnp.float32), "L": jnp.array(logit.reshape(-1), jnp.float32)}
    _, info = compute(p, p, tr, perm)
    dl = np.asarray(info["_delta"])[:N]
    return np.bincount(pid, weights=wts * dl, minlength=npairs)

with jax.default_device(jax.devices("cpu")[0]):
    jax.config.update("jax_enable_x64", False)
    true_logit = np.log(np.clip(pt, 1e-7, 1 - 1e-7) / np.clip(1 - pt, 1e-7, 1))
    e_cross = expected_delta("cross", Q, true_logit)
    chk("T1 cross: E[delta]=0 at Q* for all pairs", np.abs(e_cross).max() < 1e-4, f"max {np.abs(e_cross).max():.2e}")
    e_unif = expected_delta("unif", Q, true_logit)
    chk("T1 unif (exact head): E[delta]=0 at Q*", np.abs(e_unif).max() < 1e-4, f"max {np.abs(e_unif).max():.2e}")
    e_c0 = expected_delta("cross", Q + 7.0, true_logit)
    chk("T1 cross: invariant to a constant shift of Q", np.abs(e_c0).max() < 1e-3, f"max {np.abs(e_c0).max():.2e}")
    Qp = Q + 0.3 * rng.normal(size=Q.shape)
    chk("T2 cross: nonzero for perturbed Q", np.abs(expected_delta("cross", Qp, true_logit)).max() > 1e-2)
    chk("T2 unif: nonzero for perturbed Q", np.abs(expected_delta("unif", Qp, true_logit)).max() > 1e-2)
    bad = true_logit + 1.5 * rng.normal(size=true_logit.shape)
    eb = expected_delta("unif", Q, bad)
    chk("T3 unif with wrong head is biased at Q*", np.abs(eb).max() > 1e-2, f"max {np.abs(eb).max():.3f}")
    # T4 pair types (single deterministic rows)
    _, compute = make_pair_loss(mock, g, "cross", 0.33, 1e9, return_delta=True)
    Tq = rng.normal(size=X).astype(np.float32); p_ = {"T": jnp.array(Tq), "L": jnp.zeros(X)}
    def one(xa, ya, dx, dy, sx, sy):
        t = Transition(obs=jnp.array([xa // A, ya // A]), action=jnp.array([xa % A, ya % A]),
                       reward=jnp.array([1.3, -0.4], jnp.float32), done=jnp.array([dx, dy]),
                       next_obs=jnp.array([sx, sy]), info={})
        _, inf = compute(p_, p_, t, jnp.array([1, 0])); return float(inf["_delta"][0])
    Tm = Tq.reshape(S, A); V = lambda s: Tm[s].max()
    xa, ya, sx, sy = 4, 9, 2, 5
    Dxy = Tq[xa] - Tq[ya]
    want = 1.3 + 0.4 + g * (V(sx) - V(sy)) - Dxy
    chk("T4 both continue: r_x - r_y + g M - D", abs(one(xa, ya, False, False, sx, sy) - want) < 1e-4)
    want = ((1 - g) * 1.3 + 0.4 + g * (Tq[xa] - V(sy)) - Dxy) / (1 - g)
    chk("T4 x ends: ((1-g) r_x - r_y + g Bmin - D)/(1-g)", abs(one(xa, ya, True, False, sx, sy) - want) < 1e-3)
    want = (1.3 + 0.4 - Dxy) / (1 - g)
    chk("T4 both end: (r_x - r_y - D)/(1-g)", abs(one(xa, ya, True, True, sx, sy) - want) < 1e-3)
    # T5 PairNet
    obs = jnp.array(rng.integers(0, 2, size=(5, 10, 10, 4)), jnp.float32)
    for head in ("anti", "free"):
        net = PairNet(num_actions=A, head=head)
        prm = net.init(jax.random.PRNGKey(0), obs[:1])
        E = net.apply(prm, obs, method=PairNet.all_e)
        sc = net.apply(prm, obs, method=PairNet.scores)
        Dab = net.apply(prm, E[:, 0], E[:, 1], method=PairNet.D); Dba = net.apply(prm, E[:, 1], E[:, 0], method=PairNet.D)
        Daa = net.apply(prm, E[:, 0], E[:, 0], method=PairNet.D)
        anti = float(jnp.abs(Dab + Dba).max()); selfz = float(jnp.abs(Daa).max())
        if head == "anti":
            chk("T5 anti: shapes, D(x,y)=-D(y,x), D(x,x)=0", sc.shape == (5, A) and anti < 1e-5 and selfz < 1e-5)
        else:
            chk("T5 free: not antisymmetric at init", sc.shape == (5, A) and anti > 1e-4)
    Tq = rng.normal(size=X).astype(np.float32)
    sc = mock.apply({"T": jnp.array(Tq)}, jnp.arange(S)[:, None] * A + jnp.arange(A)[None], method=PairNet.scores_from_e)
    chk("T5 Borda argmax = argmax Q (consistent table)", np.array_equal(np.asarray(sc).argmax(1), Tq.reshape(S, A).argmax(1)))
print(f"{ok}/{tot} passed")
# T6 multi-partner: K=1 loss equals old single-shift; K>1 averages over K distinct shifts
from stoix.systems.q_learning.ff_rpair import make_pair_loss as mpl
obs = jnp.array(rng.integers(0, 2, size=(16, 10, 10, 4)), jnp.float32)
tr6 = Transition(obs=obs, action=jnp.array(rng.integers(0, A, 16)), reward=jnp.array(rng.normal(size=16), jnp.float32),
                 done=jnp.array(rng.random(16) < .3), next_obs=obs[::-1], info={})
net = PairNet(num_actions=A, head="free"); prm = net.init(jax.random.PRNGKey(1), obs[:1])
l1, c1 = mpl(net, g, "cross", .33, 1e9, n_partners=1)
lK, cK = mpl(net, g, "cross", .33, 1e9, n_partners=4)
a1 = l1(prm, prm, tr6, jnp.int32(5))[0]; b1 = c1(prm, prm, tr6, (jnp.arange(16) + 5) % 16)[0]
chk("T6 K=1 equals single shift", abs(float(a1 - b1)) < 1e-5)
aK = lK(prm, prm, tr6, jnp.int32(5))[0]
shifts = [1 + (5 - 1 + j * (15 // 4)) % 15 for j in range(4)]
bK = np.mean([float(c1(prm, prm, tr6, (jnp.arange(16) + s_) % 16)[0]) for s_ in shifts])
chk("T6 K=4 equals mean over 4 shifts (distinct, nonzero)", abs(float(aK) - bK) < 1e-4 and len(set(shifts)) == 4 and 0 not in [s_ % 16 for s_ in shifts])
# T7 sadqn: params ~ pair, loss = DQN
ns = PairNet(num_actions=A, head="sadqn"); ps = ns.init(jax.random.PRNGKey(0), obs[:1])
nf = PairNet(num_actions=A, head="free"); pf = nf.init(jax.random.PRNGKey(0), obs[:1])
cnt = lambda q: sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(q))
chk(f"T7 sadqn params {cnt(ps)} vs pair {cnt(pf)} (within 1%)", abs(cnt(ps) - cnt(pf)) / cnt(pf) < 0.01)
ls_, _ = mpl(ns, g, "cross", .33, 1e9)
qv = lambda pp, o: ns.apply(pp, o, method=PairNet.scores)
y = tr6.reward + g * (1 - tr6.done.astype(jnp.float32)) * qv(ps, tr6.next_obs).max(1)
want = 0.5 * float(jnp.mean(jnp.square(y - qv(ps, obs)[jnp.arange(16), tr6.action])))
got = float(ls_(ps, ps, tr6, jnp.int32(3))[0])
chk("T7 sadqn loss = DQN loss", abs(got - want) < 1e-5, f"{got:.5f} vs {want:.5f}")
print(f"{ok}/{tot} passed (incl. T6/T7)")
# T8-T11 (2026-10-31): lowrank head, all-pairs matrix path, rc51
nl = PairNet(num_actions=A, head="lowrank", rank=4); pl = nl.init(jax.random.PRNGKey(2), obs[:1])
E = nl.apply(pl, obs, method=PairNet.all_e)
Dfull = jnp.stack([jnp.stack([nl.apply(pl, E[:, a1], E[:, a2], method=PairNet.D) for a2 in range(A)], -1) for a1 in range(A)], -2)
chk("T8 lowrank antisymmetric", float(jnp.abs(Dfull + jnp.swapaxes(Dfull, -1, -2)).max()) < 1e-5)
borda = (Dfull.sum(-1) - jnp.einsum("baa->ba", Dfull)) / (A - 1)
chk("T8 lowrank Borda closed form = pairwise mean", float(jnp.abs(borda - nl.apply(pl, obs, method=PairNet.scores)).max()) < 1e-5)
la, _ = mpl(nl, g, "cross", .33, 1e9, all_pairs=True, return_delta=True)
_, cl = mpl(nl, g, "cross", .33, 1e9, return_delta=True)
mm = 16; ii, jj = np.meshgrid(np.arange(mm), np.arange(mm), indexing="ij"); off = ii != jj
lossA, infoA = la(pl, pl, tr6, jnp.int32(1))
lossL, infoL = cl(pl, pl, tr6, jnp.array(ii[off]), jnp.array(jj[off]))
chk("T9 all-pairs matrix loss = generic loop loss", abs(float(lossA) - float(lossL)) < 1e-5, f"{float(lossA):.6f} vs {float(lossL):.6f}")
dA = np.asarray(infoA["_delta"])[ii[off], jj[off]]
chk("T9 all-pairs delta = generic delta", np.abs(dA - np.asarray(infoL["_delta"])).max() < 1e-4)
gA = jax.grad(lambda pp: la(pp, pl, tr6, jnp.int32(1))[0])(pl); gL = jax.grad(lambda pp: cl(pp, pl, tr6, jnp.array(ii[off]), jnp.array(jj[off]))[0])(pl)
gd = max(float(jnp.abs(a_ - b_).max()) for a_, b_ in zip(jax.tree_util.tree_leaves(gA), jax.tree_util.tree_leaves(gL)))
chk("T9 all-pairs gradient = generic gradient", gd < 1e-5, f"{gd:.2e}")
nr = PairNet(num_actions=A, head="rc51", hidden=64, emb_dim=8, n_atoms=51, vmax=20.0); pr = nr.init(jax.random.PRNGKey(3), obs[:1])
lr_, cr_ = mpl(nr, g, "unif", .33, 1e9, return_delta=True)
loss_r, info_r = lr_(pr, pr, tr6, jnp.int32(3))
chk("T10 rc51 loss finite, D(x,y) is the distribution mean", bool(np.isfinite(float(loss_r))))
# mean of the mixture target equals the uniformised mean target computed from distribution means (no clipping here)
ix = jnp.arange(16); iy = (ix + 3) % 16
z = jnp.linspace(-20, 20, 51)
ap_ = lambda m_, *a: nr.apply(pr, *a, method=m_)
eo = ap_(PairNet.sa, ap_(PairNet.phi, obs), tr6.action); En = ap_(PairNet.all_e, tr6.next_obs)
ast = jnp.argmax(ap_(PairNet.scores_from_e, En), -1); en = En[jnp.arange(16), ast]
Dm = lambda a_, b_: jnp.sum(jax.nn.softmax(ap_(PairNet.rc_logits, a_, b_), -1) * z, -1)
pt = jax.nn.sigmoid(ap_(PairNet.cont_logit, eo)); rho = (1 - g) / (1 - g * (1 - pt))
d = tr6.done.astype(jnp.float32); r = tr6.reward
gx, gy = rho * (1 - d), (rho * (1 - d))[iy]
want = rho * r - rho[iy] * r[iy] + g * (gx * gy * Dm(en, en[iy]) + gx * (1 - gy) * Dm(en, eo[iy]) + (1 - gx) * gy * Dm(eo, en[iy]) + (1 - gx) * (1 - gy) * Dm(eo, eo[iy]))
got = np.asarray(info_r["_delta"]) + np.asarray(Dm(eo, eo[iy]))
chk("T11 rc51 target mean = uniformised mean target", np.abs(got - np.asarray(want)).max() < 1e-3, f"{np.abs(got - np.asarray(want)).max():.2e}")
print(f"{ok}/{tot} passed (incl. T8-T11)")
# T12-T13 (2026-11-01): MLP all-pairs via additive decomposition
for hd in ("free", "anti"):
    n0 = PairNet(num_actions=A, head=hd); n1 = PairNet(num_actions=A, head=hd, additive=True)
    pp = n0.init(jax.random.PRNGKey(5), obs[:1])
    s0 = n0.apply(pp, obs, method=PairNet.scores); s1 = n1.apply(pp, obs, method=PairNet.scores)
    chk(f"T12 {hd}: additive Borda = per-pair Borda", float(jnp.abs(s0 - s1).max()) < 1e-5, f"{float(jnp.abs(s0-s1).max()):.1e}")
    la, _ = mpl(n1, g, "cross", .33, 1e9, all_pairs=True, return_delta=True)
    _, cl = mpl(n0, g, "cross", .33, 1e9, return_delta=True)
    lossA, infoA = la(pp, pp, tr6, jnp.int32(1))
    lossL, infoL = cl(pp, pp, tr6, jnp.array(ii[off]), jnp.array(jj[off]))
    chk(f"T13 {hd}: all-pairs loss = loop loss", abs(float(lossA) - float(lossL)) < 1e-4 * max(1, abs(float(lossL))), f"{float(lossA):.6f} vs {float(lossL):.6f}")
    dA = np.asarray(infoA["_delta"])[ii[off], jj[off]]
    chk(f"T13 {hd}: all-pairs delta = loop delta", np.abs(dA - np.asarray(infoL["_delta"])).max() < 1e-3)
    gA = jax.grad(lambda q: la(q, pp, tr6, jnp.int32(1))[0])(pp); gL = jax.grad(lambda q: cl(q, pp, tr6, jnp.array(ii[off]), jnp.array(jj[off]))[0])(pp)
    gd = max(float(jnp.abs(a_ - b_).max()) for a_, b_ in zip(jax.tree_util.tree_leaves(gA), jax.tree_util.tree_leaves(gL)))
    chk(f"T13 {hd}: all-pairs gradient = loop gradient", gd < 1e-5, f"{gd:.2e}")
print(f"{ok}/{tot} passed (incl. T12-T13)")
# T14: init with additive=True works and gives the same parameter tree as the per-pair net
n1 = PairNet(num_actions=A, head="free", additive=True); pa = n1.init(jax.random.PRNGKey(5), obs[:1])
n0 = PairNet(num_actions=A, head="free"); pb = n0.init(jax.random.PRNGKey(5), obs[:1])
same = jax.tree_util.tree_structure(pa) == jax.tree_util.tree_structure(pb) and all(
    float(jnp.abs(x - y).max()) == 0 for x, y in zip(jax.tree_util.tree_leaves(pa), jax.tree_util.tree_leaves(pb)))
chk("T14 additive init: same params as per-pair init", same)
print(f"{ok}/{tot} passed (incl. T14)")
