"""(float32: value comparisons use relative tolerance 1e-6, gradients absolute 1e-5)
Checks for the counterfactual same-state pairs in ff_rpair (2026-11-02).  Run on CPU:
JAX_PLATFORMS=cpu .venv/bin/python scripts_rdqn/check_rpair_cf.py
T1  use_cf with same_coef=0: loss and gradient equal the unchanged CW-Pair loss on the real part
T2  same_loss equals the unchanged generic pair loss on a 2m-node batch with pairs (i, m+i), (m+i, i)
T3  total loss = cross + same_coef * same (value and gradient linear in same_coef)
T4  realised pair types: same-state pair with consistent D_true and both continuing -> delta = r_a - r_b + g M - D
T5  pgx env: re-stepping the same pre-step state with the same action is identical (crn); reseed_state finds the
    rng_key leaves and changes outcomes (asterix); breakout differs only through the sticky draw
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, ".")
from stoix.systems.q_learning.ff_rpair import PairNet, make_pair_loss          # noqa: E402
from stoix.systems.q_learning.ff_rdqn import CFTransition, reseed_state          # noqa: E402
from stoix.systems.q_learning.dqn_types import Transition                        # noqa: E402

ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


A, m, g = 5, 64, 0.99
key = jax.random.PRNGKey(0)
ks = jax.random.split(key, 12)
obs = (jax.random.uniform(ks[0], (m, 10, 10, 4)) < 0.2).astype(jnp.float32)
nxt = (jax.random.uniform(ks[1], (m, 10, 10, 4)) < 0.2).astype(jnp.float32)
cfn = (jax.random.uniform(ks[2], (m, 10, 10, 4)) < 0.2).astype(jnp.float32)
act = jax.random.randint(ks[3], (m,), 0, A)
b = (act + 1 + jax.random.randint(ks[4], (m,), 0, A - 1)) % A
rew = jax.random.randint(ks[5], (m,), 0, 2).astype(jnp.float32)
crew = jax.random.randint(ks[6], (m,), 0, 2).astype(jnp.float32)
done = jax.random.uniform(ks[7], (m,)) < 0.15
cdone = jax.random.uniform(ks[8], (m,)) < 0.15
info = {}
tr_cf = CFTransition(obs, act, rew, done, nxt, info, b, crew, cdone, cfn)
tr_plain = Transition(obs=obs, action=act, reward=rew, done=done, next_obs=nxt, info=info)

for head in ("free", "anti"):
    net = PairNet(num_actions=A, head=head)
    p = net.init(ks[9], obs[:1])
    tp = jax.tree_util.tree_map(lambda x: x + 0.05 * jax.random.normal(ks[10], x.shape), p)
    shift = jnp.int32(7)
    lf0, _ = make_pair_loss(net, g, "cross", 0.33, 1.0, n_partners=8)
    lfc0, _ = make_pair_loss(net, g, "cross", 0.33, 1.0, n_partners=8, use_cf=True, same_coef=0.0)
    lfc1, cmp1 = make_pair_loss(net, g, "cross", 0.33, 1.0, n_partners=8, use_cf=True, same_coef=1.0)
    lfc2, _ = make_pair_loss(net, g, "cross", 0.33, 1.0, n_partners=8, use_cf=True, same_coef=2.0)

    # T1
    (l0, i0), g0 = jax.value_and_grad(lf0, has_aux=True)(p, tp, tr_plain, shift)
    (lc0, ic0), gc0 = jax.value_and_grad(lfc0, has_aux=True)(p, tp, tr_cf, shift)
    gd = max(float(jnp.max(jnp.abs(a - c))) for a, c in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(gc0)))
    report(f"T1[{head}] same_coef=0 == CW-Pair", abs(float(l0 - lc0)) < 1e-6 * abs(float(l0)) and gd < 1e-5,
           f"loss {float(l0):.6f} vs {float(lc0):.6f}, max grad diff {gd:.2e}")

    # T2: reference = generic pair path on 2m nodes
    obs2 = jnp.concatenate([obs, obs]); act2 = jnp.concatenate([act, b]); rew2 = jnp.concatenate([rew, crew])
    done2 = jnp.concatenate([done, cdone]); nxt2 = jnp.concatenate([nxt, cfn])
    tr2 = Transition(obs=obs2, action=act2, reward=rew2, done=done2, next_obs=nxt2, info=info)
    _, cmp_ref = make_pair_loss(net, g, "cross", 0.33, 1.0)
    ar = jnp.arange(m)
    lref, iref = cmp_ref(p, tp, tr2, jnp.concatenate([ar, ar + m]), jnp.concatenate([ar + m, ar]))
    (l1, i1) = lfc1(p, tp, tr_cf, shift)
    report(f"T2[{head}] same_loss == generic loss on (i, m+i)", abs(float(i1["same_loss"] - lref)) < 1e-5,
           f"{float(i1['same_loss']):.6f} vs {float(lref):.6f}")

    # T3
    (l2, i2), g2 = jax.value_and_grad(lfc2, has_aux=True)(p, tp, tr_cf, shift)
    (l1b, _), g1 = jax.value_and_grad(lfc1, has_aux=True)(p, tp, tr_cf, shift)
    lin = abs(float(l2 - 2 * l1b + lc0)) < 1e-6 * abs(float(l2))
    gl = max(float(jnp.max(jnp.abs(a - 2 * c + d))) for a, c, d in zip(*(jax.tree_util.tree_leaves(x) for x in (g2, g1, gc0))))
    report(f"T3[{head}] linear in same_coef", lin and gl < 1e-5, f"value resid {float(l2 - 2 * l1b + lc0):.2e}, grad resid {gl:.2e}")

# T4: realised pair-type formula for a both-continue same-state pair (cross weighting reduces to (1-g)[..]/(1-g))
net = PairNet(num_actions=A, head="free"); p = net.init(ks[9], obs[:1])
_, cmp1 = make_pair_loss(net, g, "cross", 0.33, 1.0, n_partners=8, use_cf=True, return_delta=True)
tr_nc = CFTransition(obs, act, rew, jnp.zeros_like(done), nxt, info, b, crew, jnp.zeros_like(cdone), cfn)
_, inf = cmp1(p, p, tr_nc, jnp.arange(m), (jnp.arange(m) + 1) % m)
ap = lambda method, *a: net.apply(p, *a, method=method)  # noqa: E731
phi = ap(PairNet.phi, obs)
ea, eb = ap(PairNet.sa, phi, act), ap(PairNet.sa, phi, b)
En_a, En_b = ap(PairNet.all_e, nxt), ap(PairNet.all_e, cfn)
sa_ = jnp.argmax(ap(PairNet.scores_from_e, En_a), -1); sb_ = jnp.argmax(ap(PairNet.scores_from_e, En_b), -1)
Ma = ap(PairNet.D, En_a[jnp.arange(m), sa_], En_b[jnp.arange(m), sb_])
want = rew - crew + g * Ma - ap(PairNet.D, ea, eb)
got = inf["_delta_same"][:m]
report("T4 both-continue same-state delta = r_a - r_b + g M - D", float(jnp.max(jnp.abs(got - want))) < 1e-4,
       f"max err {float(jnp.max(jnp.abs(got - want))):.2e}")

# T5: environments (pgx through the same wrappers as training)
from hydra import compose, initialize_config_dir                                  # noqa: E402
import os                                                                          # noqa: E402
from omegaconf import OmegaConf                                                    # noqa: E402
from stoix.utils import make_env as environments                                   # noqa: E402

cdir = os.path.abspath("stoix/configs/default/anakin")
for game in ("asterix", "breakout"):
    with initialize_config_dir(config_dir=cdir, version_base="1.2"):
        cfg = compose(config_name="default_ff_rdqn.yaml", overrides=[f"env=pgx/{game}", "env.wrapper=null",
                                                                       "arch.total_num_envs=128"])
    OmegaConf.set_struct(cfg, False)
    env, _ = environments.make(cfg)
    n = 256
    st, ts = env.reset(jax.random.split(jax.random.PRNGKey(1), n))
    nA = int(env.action_space().num_values)
    rk = jax.random.PRNGKey(2)
    same_det, diff_ind, steps = 0, 0, 0
    found = []
    jax.tree_util.tree_map_with_path(lambda pth, x: found.append(1) if (getattr(pth[-1], "name", None) or getattr(pth[-1], "key", None)) == "rng_key" else None, st)
    for t in range(60):
        rk, k1, k2 = jax.random.split(rk, 3)
        a = jax.random.randint(k1, (n,), 0, nA)
        s1, t1 = env.step(st, a)
        _, t2 = env.step(st, a)
        _, t3 = env.step(reseed_state(st, k2), a)
        o1, o2, o3 = t1.extras["next_obs"], t2.extras["next_obs"], t3.extras["next_obs"]
        same_det += int(jnp.sum(jnp.all((o1 == o2).reshape(n, -1), -1) & (t1.reward == t2.reward)))
        diff_ind += int(jnp.sum(~(jnp.all((o1 == o3).reshape(n, -1), -1) & (t1.reward == t3.reward))))
        steps += n
        st = s1
    report(f"T5[{game}] crn re-step identical", same_det == steps, f"{same_det}/{steps}; rng_key leaves found: {len(found)}")
    print(f"      [{game}] ind: outcome differs in {diff_ind / steps:.1%} of steps (same action)")
    if game == "asterix":
        report("T5[asterix] ind changes outcomes", diff_ind > 0)

print("ALL PASS" if ok_all else "SOME FAILED")
