"""check reset_continuing step: done stored 0; at real terminations next_obs == next transition's obs (reset obs)
and differs from the terminal obs; non-terminal steps unchanged."""
import sys, jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, "/home/junweiluo/Stoix")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stoix.systems.q_learning import ff_rdqn as M
from stoix.utils import make_env as environments
from stoix.utils.total_timestep_checker import check_total_timesteps
game = sys.argv[1]; ok = tot = 0
def chk(n, c):
    global ok, tot; tot += 1; ok += bool(c); print(("PASS " if c else "FAIL ") + n)
for reset in (True, False):
    with initialize_config_dir(config_dir="/home/junweiluo/Stoix/stoix/configs/default/anakin", version_base="1.2"):
        cfg = compose("default_ff_rdqn.yaml", overrides=["env.wrapper=null", "network=minatar_dqn", f"env=gymnax/{game}",
                      f"system.reset_continuing={reset}", "arch.total_num_envs=16"])
    OmegaConf.set_struct(cfg, False); cfg.num_devices = 1; cfg = check_total_timesteps(cfg)
    env, _ = environments.make(config=cfg)
    cfg.system.action_dim = int(env.action_space().num_values)
    import hydra
    from stoix.networks.base import FeedForwardActor as Actor
    torso = hydra.utils.instantiate(cfg.network.actor_network.pre_torso)
    head = hydra.utils.instantiate(cfg.network.actor_network.action_head, action_dim=cfg.system.action_dim, epsilon=1.0)
    net = Actor(torso=torso, action_head=head)
    x0 = env.observation_space().generate_value()[None]
    p = {"q": net.init(jax.random.PRNGKey(0), x0)}
    step = jax.jit(M.make_step_fn(env, lambda pp, o: net.apply(pp["q"], o), cfg))
    st, ts = env.reset(jax.random.split(jax.random.PRNGKey(1), 16))
    key = jax.random.PRNGKey(2); trs = []; terms = []
    for t in range(1500 if game != "freeway" else 2600):
        st, ts2, key, tr = step(p, st, ts, key, jnp.int32(10**7))
        trs.append(jax.tree_util.tree_map(np.asarray, tr)); terms.append(np.asarray(ts2.last()).reshape(-1))
        term_obs = np.asarray(ts2.extras["next_obs"]); ts = ts2
        trs[-1] = (trs[-1], term_obs)
    D = np.array([np.asarray(tr.done) for tr, _ in trs]); T = np.array(terms)
    nterm = int(T.sum()); print(f"reset={reset}: terminations {nterm}")
    if reset:
        chk("done stored as 0 everywhere", D.sum() == 0)
        m1 = m2 = c = 0
        for t in range(len(trs) - 1):
            for e in np.where(T[t])[0]:
                c += 1
                m1 += np.array_equal(trs[t][0].next_obs[e], trs[t + 1][0].obs[e])
                m2 += not np.array_equal(trs[t][0].next_obs[e], trs[t][1][e])
        chk(f"terminal next_obs == next transition obs ({m1}/{c})", c > 0 and m1 == c)
        chk(f"terminal next_obs != terminal obs ({m2}/{c})", m2 >= 0.9 * c)
        nt = ~T
        same = all(np.array_equal(trs[t][0].next_obs[e], trs[t][1][e]) for t in range(0, len(trs), 50) for e in np.where(nt[t])[0][:4])
        chk("non-terminal next_obs == env next_obs", same)
    else:
        chk("plain mode: done == termination", np.array_equal(D.astype(bool), T))
print(f"{ok}/{tot} passed")
