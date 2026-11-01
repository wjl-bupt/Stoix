"""Pairwise mean critic for MinAtar (Anakin, single file).  2026-10-31.

D(x, y) approximates Q(x) - Q(y) for state-action pairs x = (s, a), y = (s', a').
Network (PairNet):
  phi(s)   = MinAtarTorso(s)                                  (Conv16 3x3 VALID -> Dense128)
  e(s, a)  = Dense128(relu(Dense128([phi(s), emb(a)])))      (emb: 16-d action embedding)
  g(ex,ey) = Dense1(relu(Dense128([ex, ey])))
  head=free : D(x, y) = g(e_x, e_y)                           (R-C51 style, no constraint)
  head=anti : D(x, y) = g(e_x, e_y) - g(e_y, e_x)            (antisymmetric, D(x, x) = 0)
Action read-out (actor, target greedy action, evaluation): Borda score(s, a) = mean_{b != a} D((s,a), (s,b)).
Pairing (system.n_partners = K, default 1): each transition is x in K pairs with K distinct cyclic shifts.
head=lowrank (2026-10-31): D = f(x)-f(y)+z(x).z'(y)-z(y).z'(x), f=w^T h, z=Z1 h, z'=Z2 h (rank system.pair_rank);
  Borda read-out in closed form; with system.all_pairs=True all m(m-1) ordered pairs of the batch are used, computed as
  m x m matrices (one matmul each).  See 02_方法/交叉加权成对critic_算法描述_20261031.
head=free/anti with system.all_pairs=True (2026-11-01): all m(m-1) ordered pairs with the SAME MLP head, computed by the
  additive decomposition Dense([ex,ey]) = ex K1 + ey K2 + b (Bahdanau-style): U, V once per transition, then
  relu(U_i + V_j) . w2 + b2 for every pair (no per-pair matmul); Borda read-out uses the same decomposition.
head=rc51 (2026-10-31): Relative C51 (R-C51) reproduced from its appendix C on the MinAtar body: (s,a) encoder
  [phi(s), 48-d action embedding] -> 2 x Dense512; pair head [e_x, e_y] -> 2 x Dense512 -> rc_atoms logits over
  [-rc_vmax, rc_vmax]; continuation head (BCE, cont_coef); Bellman uniformization with gbar = gamma, four-way stay/go
  mixture of projected targets; one partner per transition (cyclic shift); greedy actions by Borda score of the mean.
head=sadqn: same-capacity DQN control, Q(s,a) = Dense1(relu(Dense256(e(s,a)))), standard DQN loss on the same
encoder (pair head and continuation head not created).
Old single-partner description: a batch of m replay transitions; transition i is paired with transition (i + k) mod m, k ~ U{1..m-1}
(one random shift per update; every pair has two different transitions; each transition is once x and once y).

Bootstrap terms (target network D-, greedy actions a*_x, a*_y at s'_x, s'_y by the target Borda score):
  M     = D-((s'_x,a*_x), (s'_y,a*_y))      Amax = D-((s'_x,a*_x), y)
  Bmin  = D-(x, (s'_y,a*_y))                Dst  = D-(x, y)
Realised Gamma = g(1-d), v = 1 - Gamma.

term=cross (head-free cross weighting, our method).  With k = (1-g)/g:
  xi = v_y r_x - v_x r_y + W1 M + W2 Amax + W3 Bmin + W4 D(x,y)
  W1 = k Gx Gy,  W2 = Gx (v_y - k Gy),  W3 = Gy (v_x - k Gx),  W4 = -v_y - Gy (v_x - k Gx)
  xi is linear in the realised outcomes of two independent transitions, so E[xi | x, y] =
  vbar_x vbar_y (z(x) - z(y)) with z = e/vbar: its zero set (consistent D) is exactly Q(x) - Q(y), no continuation
  head.  Pair types (realised): both continue -> xi = (1-g)[r_x - r_y + g M - D];  x ends, y continues ->
  xi = (1-g) r_x - r_y + g Bmin - D;  both end -> xi = r_x - r_y - D.  Semi-gradient step D += lr * xi/(1-g).
term=unif (R-C51 Bellman uniformization, mean version, Rao-Blackwellised over stay/go, gbar = g):
  continuation head p(x) = sigmoid(Dense1(e_x)), BCE on d with coefficient cont_coef (0.33);
  rho = (1-g)/vhat, vhat = 1 - g(1 - p-(x)) (target-network head), go prob gg = rho (1-d)
  target = rho_x r_x - rho_y r_y + g[gg_x gg_y M + gg_x(1-gg_y) Amax + (1-gg_x) gg_y Bmin + (1-gg_x)(1-gg_y) Dst]
  delta = target - D(x,y)
Loss 0.5 * mean((sg[D + delta] - D)^2)  (semi-gradient; delta = xi/(1-g) for cross).  Adam + global clip,
hard target sync every target_period updates.  Same MinAtar protocol as ff_rdqn (minatar_base).
"""
import copy
import json
import os
import time
from typing import Any, Dict

import chex
import distrax
import flashbax as fbx
import flax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf
from stoa import get_final_step_metrics

from stoix.base_types import AnakinExperimentOutput, OffPolicyLearnerState, OnlineAndTarget
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.systems.q_learning.dqn_types import Transition
from stoix.systems.q_learning.ff_rdqn import MinAtarTorso, get_warmup_fn, make_step_fn
from stoix.utils import make_env as environments
from stoix.utils.jax_utils import unreplicate_batch_dim
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate


class PairNet(nn.Module):
    num_actions: int
    head: str = "anti"      # anti | free | sadqn | lowrank | rc51
    hidden: int = 128
    emb_dim: int = 16
    rank: int = 32          # lowrank: rank r_b of the antisymmetric interaction
    additive: bool = False  # free/anti: additive decomposition of the first pair layer (all-pairs + Borda), same function
    n_atoms: int = 51       # rc51: atoms of the return-difference distribution
    vmax: float = 20.0      # rc51: support [-vmax, vmax]

    def setup(self):
        self.torso = MinAtarTorso()
        self.aemb = nn.Embed(self.num_actions, self.emb_dim)
        self.sa1 = nn.Dense(self.hidden)
        self.sa2 = nn.Dense(self.hidden)
        self.p1 = nn.Dense(self.hidden)
        self.p1b = nn.Dense(self.hidden)        # rc51 only (second hidden layer of the pair head)
        self.p2 = nn.Dense(self.n_atoms if self.head == "rc51" else 1)
        self.lr_f = nn.Dense(1)                 # lowrank: f = w^T h
        self.lr_z1 = nn.Dense(self.rank, use_bias=False)
        self.lr_z2 = nn.Dense(self.rank, use_bias=False)
        self.q1 = nn.Dense(2 * self.hidden)     # sadqn only: 128*256+256+257 params ~= pair head 256*128+128+129
        self.q2 = nn.Dense(1)
        self.cont = nn.Dense(1)

    def phi(self, obs):
        return self.torso(obs)

    def sa(self, phi, a):
        h = jnp.concatenate([phi, self.aemb(a)], axis=-1)
        return self.sa2(nn.relu(self.sa1(h)))

    def g(self, ex, ey):
        return self.p2(nn.relu(self.p1(jnp.concatenate([ex, ey], axis=-1))))[..., 0]

    def comp(self, e):
        """lowrank components (f, z, z') of one (s,a) encoding."""
        return self.lr_f(e)[..., 0], self.lr_z1(e), self.lr_z2(e)

    def rc_logits(self, ex, ey):
        h = nn.relu(self.p1(jnp.concatenate([ex, ey], axis=-1)))
        return self.p2(nn.relu(self.p1b(h)))                            # [..., n_atoms]

    def support(self):
        return jnp.linspace(-self.vmax, self.vmax, self.n_atoms)

    def qval(self, e):
        return self.q2(nn.relu(self.q1(e)))[..., 0]

    def D(self, ex, ey):
        if self.head == "sadqn":
            return self.qval(ex) - self.qval(ey)
        if self.head == "lowrank":
            fx, zx, zpx = self.comp(ex); fy, zy, zpy = self.comp(ey)
            return fx - fy + jnp.sum(zx * zpy, -1) - jnp.sum(zy * zpx, -1)
        if self.head == "rc51":
            return jnp.sum(jax.nn.softmax(self.rc_logits(ex, ey), -1) * self.support(), -1)
        if self.head == "anti":
            return self.g(ex, ey) - self.g(ey, ex)
        return self.g(ex, ey)

    def all_e(self, obs):
        phi = self.phi(obs)                                             # [B, F]
        B = phi.shape[0]
        a = jnp.broadcast_to(jnp.arange(self.num_actions), (B, self.num_actions))
        phib = jnp.broadcast_to(phi[:, None, :], (B, self.num_actions, phi.shape[-1]))
        return self.sa(phib, a)                                         # [B, A, E]

    def pre(self, e):
        """Additive decomposition of the first pair layer: Dense([ex, ey]) = ex K1 + ey K2 + b.
        Returns (U, V) with U = e K1 (x-role), V = e K2 + b (y-role)."""
        Kb = self.p1.variables["params"]
        K, b = Kb["kernel"], Kb["bias"]
        dh = e.shape[-1]
        return e @ K[:dh], e @ K[dh:] + b

    def g_add(self, U, V):
        """g for every combination: U [..., n, H], V [..., k, H] -> [..., n, k]."""
        h = nn.relu(U[..., :, None, :] + V[..., None, :, :])
        Kb = self.p2.variables["params"]
        return (h @ Kb["kernel"])[..., 0] + Kb["bias"][0]

    def Dmat(self, ex, ey):
        """D(ex_i, ey_j) for all i, j (free / anti MLP head, additive form)."""
        Ux, Vx = self.pre(ex); Uy, Vy = self.pre(ey)
        G = self.g_add(Ux, Vy)
        if self.head == "anti":
            G = G - jnp.swapaxes(self.g_add(Uy, Vx), -1, -2)
        return G

    def scores_from_e(self, E):
        if self.additive and self.head in ("free", "anti"):            # same Borda, additive computation
            A = self.num_actions
            Dm = self.Dmat(E, E)                                        # [B, A, A]
            off = 1.0 - jnp.eye(A)
            return (Dm * off).sum(-1) / (A - 1)
        if self.head == "sadqn":
            return self.qval(E)
        if self.head == "lowrank":                                     # Borda closed form, O(A r) per state
            A = self.num_actions
            f, z, zp = self.comp(E)                                     # [B,A], [B,A,r], [B,A,r]
            Sf = f.sum(-1, keepdims=True); Sz = z.sum(-2, keepdims=True); Szp = zp.sum(-2, keepdims=True)
            return (A * f - Sf + jnp.sum(z * Szp, -1) - jnp.sum(Sz * zp, -1)) / (A - 1)
        A = self.num_actions
        ex = jnp.broadcast_to(E[:, :, None, :], (E.shape[0], A, A, E.shape[-1]))
        ey = jnp.broadcast_to(E[:, None, :, :], (E.shape[0], A, A, E.shape[-1]))
        Dm = self.D(ex, ey)                                             # [B, A, A], Dm[a, b] = D((s,a),(s,b))
        off = 1.0 - jnp.eye(A)
        return (Dm * off).sum(-1) / (A - 1)

    def scores(self, obs):
        return self.scores_from_e(self.all_e(obs))

    def cont_logit(self, e):
        return self.cont(e)[..., 0]

    def __call__(self, obs):                                            # init: touch the used submodules
        E = self.all_e(obs)
        if self.head == "sadqn":
            return self.scores_from_e(E)
        if self.additive:                                               # create p1/p2 before the additive path reads them
            self.g(E[:, 0], E[:, 0])
        return self.scores_from_e(E), self.cont_logit(E[:, 0])


def make_pair_loss(net, gamma: float, term: str, cont_coef: float, max_abs_reward: float,
                   return_delta: bool = False, n_partners: int = 1, all_pairs: bool = False):
    k = (1.0 - gamma) / gamma

    def ap(p, method, *args):
        return net.apply(p, *args, method=method)

    def pairmat(c1, c2):
        """D(u_i, u'_j) for all i, j from lowrank components c1 = comp(u), c2 = comp(u')."""
        f1, z1, zp1 = c1; f2, z2, zp2 = c2
        return f1[:, None] - f2[None, :] + z1 @ zp2.T - zp1 @ z2.T

    def lowrank_all_pairs(p, tp, r, d, e_on, e_t, e_n):
        m = r.shape[0]
        c_on = ap(p, PairNet.comp, e_on); c_t = ap(tp, PairNet.comp, e_t); c_n = ap(tp, PairNet.comp, e_n)
        Dm = pairmat(c_on, c_on)
        M, Amx, Bmn = pairmat(c_n, c_n), pairmat(c_n, c_t), pairmat(c_t, c_n)
        Gx, Gy = (gamma * (1 - d))[:, None], (gamma * (1 - d))[None, :]
        vx, vy = 1 - Gx, 1 - Gy
        W1 = k * Gx * Gy; W2 = Gx * (vy - k * Gy); W3 = Gy * (vx - k * Gx); W4 = -vy - Gy * (vx - k * Gx)
        xi = vy * r[:, None] - vx * r[None, :] + W1 * M + W2 * Amx + W3 * Bmn + W4 * jax.lax.stop_gradient(Dm)
        delta = jax.lax.stop_gradient(xi) / (1.0 - gamma)
        mask = 1.0 - jnp.eye(m)
        npair = m * (m - 1)
        loss = 0.5 * jnp.sum(mask * jnp.square(jax.lax.stop_gradient(Dm + delta) - Dm)) / npair
        dx, dy = d[:, None], d[None, :]
        one_term = mask * jnp.clip(dx + dy, 0, 1) * (1 - dx * dy)
        nn_ = mask * (1 - dx) * (1 - dy)
        sq = jnp.square(delta) * mask
        info = {"q_loss": loss, "td_std": jnp.sqrt(jnp.sum(sq) / npair), "delta_abs": jnp.sum(jnp.abs(delta) * mask) / npair,
                "delta_abs_nn": jnp.sum(jnp.abs(delta) * nn_) / jnp.maximum(jnp.sum(nn_), 1.0),
                "delta_abs_oneterm": jnp.sum(jnp.abs(delta) * one_term) / jnp.maximum(jnp.sum(one_term), 1.0),
                "oneterm_pair_frac": jnp.sum(one_term) / npair,
                "oneterm_sq_share": jnp.sum(sq * one_term) / jnp.maximum(jnp.sum(sq), 1e-12),
                "term_frac": jnp.mean(d), "D_abs": jnp.sum(jnp.abs(Dm) * mask) / npair,
                "skew_abs": jnp.sum(jnp.abs(c_on[1] @ c_on[2].T - c_on[2] @ c_on[1].T) * mask) / npair}
        if return_delta:
            info["_delta"] = delta
        return loss, info

    def mlp_all_pairs(p, tp, r, d, e_on, e_t, e_n):
        """free / anti MLP head, all m(m-1) ordered pairs, additive decomposition (no per-pair matmul)."""
        m = r.shape[0]
        Dm = ap(p, PairNet.Dmat, e_on, e_on)
        M, Amx, Bmn = ap(tp, PairNet.Dmat, e_n, e_n), ap(tp, PairNet.Dmat, e_n, e_t), ap(tp, PairNet.Dmat, e_t, e_n)
        Gx, Gy = (gamma * (1 - d))[:, None], (gamma * (1 - d))[None, :]
        vx, vy = 1 - Gx, 1 - Gy
        W1 = k * Gx * Gy; W2 = Gx * (vy - k * Gy); W3 = Gy * (vx - k * Gx); W4 = -vy - Gy * (vx - k * Gx)
        xi = vy * r[:, None] - vx * r[None, :] + W1 * M + W2 * Amx + W3 * Bmn + W4 * jax.lax.stop_gradient(Dm)
        delta = jax.lax.stop_gradient(xi) / (1.0 - gamma)
        mask = 1.0 - jnp.eye(m)
        npair = m * (m - 1)
        loss = 0.5 * jnp.sum(mask * jnp.square(jax.lax.stop_gradient(Dm + delta) - Dm)) / npair
        dx, dy = d[:, None], d[None, :]
        one_term = mask * jnp.clip(dx + dy, 0, 1) * (1 - dx * dy)
        nn_ = mask * (1 - dx) * (1 - dy)
        sq = jnp.square(delta) * mask
        Dsg = jax.lax.stop_gradient(Dm)
        info = {"q_loss": loss, "td_std": jnp.sqrt(jnp.sum(sq) / npair), "delta_abs": jnp.sum(jnp.abs(delta) * mask) / npair,
                "delta_abs_nn": jnp.sum(jnp.abs(delta) * nn_) / jnp.maximum(jnp.sum(nn_), 1.0),
                "delta_abs_oneterm": jnp.sum(jnp.abs(delta) * one_term) / jnp.maximum(jnp.sum(one_term), 1.0),
                "oneterm_pair_frac": jnp.sum(one_term) / npair,
                "oneterm_sq_share": jnp.sum(sq * one_term) / jnp.maximum(jnp.sum(sq), 1e-12),
                "term_frac": jnp.mean(d), "D_abs": jnp.sum(jnp.abs(Dsg) * mask) / npair,
                "antisym_gap": jnp.sum(jnp.abs(Dsg + Dsg.T) * mask) / npair}
        if return_delta:
            info["_delta"] = delta
        return loss, info

    def rc51_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all, ix, iy):
        """R-C51 (mean-preserving Bellman uniformization, four-way stay/go mixture, categorical projection)."""
        from stoix.systems.q_learning.ff_rdqn import c51_project
        z = jnp.linspace(-net.vmax, net.vmax, net.n_atoms)
        rx, ry = r_all[ix], r_all[iy]; dx, dy = d_all[ix], d_all[iy]
        logit_on = ap(p, PairNet.rc_logits, e_on_all[ix], e_on_all[iy])
        p_t = jax.nn.sigmoid(ap(tp, PairNet.cont_logit, e_t_all))      # continuation-head estimate of P(done)
        vhat = 1.0 - gamma * (1.0 - p_t)
        rho_all = (1.0 - gamma) / vhat
        rho_x, rho_y = rho_all[ix], rho_all[iy]
        gx, gy = rho_x * (1 - dx), rho_y * (1 - dy)                     # go probabilities (gbar = gamma)
        rew = rho_x * rx - rho_y * ry
        disc = jnp.full_like(rew, gamma)
        sm = lambda a_, b_: jax.nn.softmax(ap(tp, PairNet.rc_logits, a_, b_), -1)   # noqa: E731
        ex_g, ex_s = e_n_all[ix], e_t_all[ix]; ey_g, ey_s = e_n_all[iy], e_t_all[iy]
        tgt = (gx * gy)[:, None] * c51_project(sm(ex_g, ey_g), rew, disc, z) \
            + (gx * (1 - gy))[:, None] * c51_project(sm(ex_g, ey_s), rew, disc, z) \
            + ((1 - gx) * gy)[:, None] * c51_project(sm(ex_s, ey_g), rew, disc, z) \
            + ((1 - gx) * (1 - gy))[:, None] * c51_project(sm(ex_s, ey_s), rew, disc, z)
        tgt = jax.lax.stop_gradient(tgt)
        ce = -jnp.mean(jnp.sum(tgt * jax.nn.log_softmax(logit_on, -1), -1))
        logit_c = ap(p, PairNet.cont_logit, e_on_all)
        bce = optax.sigmoid_binary_cross_entropy(logit_c, d_all).mean()
        D_mean = jnp.sum(jax.nn.softmax(logit_on, -1) * z, -1)
        delta = jax.lax.stop_gradient(jnp.sum(tgt * z, -1) - D_mean)
        pc = jax.nn.sigmoid(logit_c)
        info = {"q_loss": ce, "td_std": jnp.std(delta), "delta_abs": jnp.mean(jnp.abs(delta)),
                "term_frac": jnp.mean(d_all), "D_abs": jnp.mean(jnp.abs(D_mean)), "bce": bce,
                "p_hat_term": jnp.sum(pc * d_all) / jnp.maximum(d_all.sum(), 1.0),
                "p_hat_nonterm": jnp.sum(pc * (1 - d_all)) / jnp.maximum((1 - d_all).sum(), 1.0),
                "mass_at_edges": jnp.mean(tgt[:, 0] + tgt[:, -1])}
        if return_delta:
            info["_delta"] = delta
        return ce + cont_coef * bce, info

    def loss_fn(p, tp, tr, shift):
        """n_partners distinct cyclic shifts: every transition is x in n_partners pairs (unbiased, lower variance)."""
        m = tr.action.shape[0]
        K = n_partners
        shifts = 1 + (shift - 1 + jnp.arange(K) * ((m - 1) // K)) % (m - 1)
        ix = jnp.tile(jnp.arange(m), K)
        iy = ((jnp.arange(m)[None, :] + shifts[:, None]) % m).reshape(-1)
        return compute(p, tp, tr, ix, iy)

    def compute(p, tp, tr, ix, iy=None):
        """pairs (ix[j], iy[j]); with a single argument it is the old interface (ix = perm, x = arange)."""
        if iy is None:
            iy = ix; ix = jnp.arange(tr.action.shape[0])
        obs, act, nxt = tr.obs, tr.action, tr.next_obs
        r_all = jnp.clip(tr.reward, -max_abs_reward, max_abs_reward).astype(jnp.float32)
        d_all = tr.done.astype(jnp.float32)
        m = act.shape[0]
        # per-transition encodings (computed once, gathered per pair)
        e_on_all = ap(p, PairNet.sa, ap(p, PairNet.phi, obs), act)      # [m, E]
        e_t_all = ap(tp, PairNet.sa, ap(tp, PairNet.phi, obs), act)
        En = ap(tp, PairNet.all_e, nxt)                                  # [m, A, E]
        a_star = jnp.argmax(ap(tp, PairNet.scores_from_e, En), axis=-1)
        e_n_all = En[jnp.arange(m), a_star]                              # [m, E]
        info = {}
        if net.head == "sadqn":                                         # plain DQN on the same encoder
            q_on = ap(p, PairNet.qval, e_on_all)
            y = r_all + gamma * (1.0 - d_all) * ap(tp, PairNet.qval, e_n_all)
            delta = jax.lax.stop_gradient(y - q_on)
            loss = 0.5 * jnp.mean(jnp.square(delta))
            info.update({"q_loss": loss, "td_std": jnp.std(delta), "delta_abs": jnp.mean(jnp.abs(delta)),
                         "term_frac": jnp.mean(d_all), "q_tilde_mean": jnp.mean(q_on),
                         "D_abs": jnp.mean(jnp.abs(q_on - q_on[jnp.roll(jnp.arange(m), 1)]))})
            if return_delta:
                info["_delta"] = delta
            return 0.5 * jnp.mean(jnp.square(jax.lax.stop_gradient(q_on + delta) - q_on)), info
        if net.head == "lowrank" and all_pairs:
            return lowrank_all_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all)
        if net.head in ("free", "anti") and all_pairs:
            return mlp_all_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all)
        if net.head == "rc51":
            return rc51_pairs(p, tp, r_all, d_all, e_on_all, e_t_all, e_n_all, ix, iy)
        e_on, e_t, e_n = e_on_all[ix], e_t_all[ix], e_n_all[ix]
        e_on_y, e_t_y, e_n_y = e_on_all[iy], e_t_all[iy], e_n_all[iy]
        D_xy = ap(p, PairNet.D, e_on, e_on_y)
        M = ap(tp, PairNet.D, e_n, e_n_y)
        Amax = ap(tp, PairNet.D, e_n, e_t_y)
        Bmin = ap(tp, PairNet.D, e_t, e_n_y)
        Dst = ap(tp, PairNet.D, e_t, e_t_y)
        r, d = r_all[ix], d_all[ix]
        rx, ry = r, r_all[iy]
        dy = d_all[iy]
        Gx, Gy = gamma * (1.0 - d), gamma * (1.0 - dy)
        vx, vy = 1.0 - Gx, 1.0 - Gy
        info = {}
        if term == "cross":
            W1 = k * Gx * Gy
            W2 = Gx * (vy - k * Gy)
            W3 = Gy * (vx - k * Gx)
            W4 = -vy - Gy * (vx - k * Gx)
            xi = vy * rx - vx * ry + W1 * M + W2 * Amax + W3 * Bmin + W4 * D_xy
            delta = jax.lax.stop_gradient(xi) / (1.0 - gamma)
            aux = 0.0
        else:
            p_t = jax.nn.sigmoid(ap(tp, PairNet.cont_logit, e_t_all))
            vhat = 1.0 - gamma * (1.0 - p_t)
            rho_all = (1.0 - gamma) / vhat
            rho, rho_y = rho_all[ix], rho_all[iy]
            ggx, ggy = rho * (1.0 - d), rho_y * (1.0 - dy)
            target = rho * rx - rho_y * ry + gamma * (ggx * ggy * M + ggx * (1 - ggy) * Amax
                                                          + (1 - ggx) * ggy * Bmin + (1 - ggx) * (1 - ggy) * Dst)
            delta = jax.lax.stop_gradient(target - D_xy)
            logit = ap(p, PairNet.cont_logit, e_on_all)
            d = d_all
            bce = optax.sigmoid_binary_cross_entropy(logit, d).mean()
            aux = cont_coef * bce
            pc = jax.nn.sigmoid(logit)
            nt = jnp.maximum(d.sum(), 1.0)
            info["p_hat_term"] = (pc * d).sum() / nt
            info["p_hat_nonterm"] = (pc * (1 - d)).sum() / jnp.maximum((1 - d).sum(), 1.0)
            info["bce"] = bce
        tgt = jax.lax.stop_gradient(D_xy + delta)
        pair_loss = 0.5 * jnp.mean(jnp.square(tgt - D_xy))
        d = d_all[ix]
        one_term = jnp.clip(d + dy, 0, 1) * (1 - d * dy)
        sq = jnp.square(delta)
        info.update({
            "q_loss": pair_loss, "td_std": jnp.std(delta), "delta_abs": jnp.mean(jnp.abs(delta)),
            "delta_abs_nn": jnp.sum(jnp.abs(delta) * (1 - d) * (1 - dy)) / jnp.maximum(jnp.sum((1 - d) * (1 - dy)), 1.0),
            "delta_abs_oneterm": jnp.sum(jnp.abs(delta) * one_term) / jnp.maximum(jnp.sum(one_term), 1.0),
            "oneterm_pair_frac": jnp.mean(one_term),
            "oneterm_sq_share": jnp.sum(sq * one_term) / jnp.maximum(jnp.sum(sq), 1e-12),
            "term_frac": jnp.mean(d_all), "D_abs": jnp.mean(jnp.abs(D_xy)),
        })
        if return_delta:
            info["_delta"] = delta
        return pair_loss + aux, info

    return loss_fn, compute


def get_learner_fn(env, net, q_apply, q_update_fn, buffer_fns, config):
    buffer_add_fn, buffer_sample_fn = buffer_fns
    step = make_step_fn(env, q_apply, config)
    loss_fn, _ = make_pair_loss(net, float(config.system.gamma), str(config.system.term),
                                float(config.system.cont_coef), float(config.system.max_abs_reward),
                                n_partners=int(config.system.get("n_partners", 1)),
                                all_pairs=bool(config.system.get("all_pairs", False)))
    target_period = int(config.system.target_period)
    env_steps_per_update = int(config.system.rollout_length) * int(config.arch.num_envs)
    m = int(config.system.batch_size)

    def _update_step(learner_state, _):
        def _env_step(learner_state, _):
            q_params, opt_states, buffer_state, key, env_state, last_timestep = learner_state
            env_state, timestep, key, tr = step(q_params.online, env_state, last_timestep, key, opt_states["t"])
            return OffPolicyLearnerState(q_params, opt_states, buffer_state, key, env_state, timestep), tr

        learner_state, traj_batch = jax.lax.scan(_env_step, learner_state, None, config.system.rollout_length)
        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state
        buffer_state = buffer_add_fn(buffer_state, traj_batch)
        opt_states = {**opt_states, "t": opt_states["t"] + env_steps_per_update}

        def _update_epoch(update_state, _):
            params, opt_states, buffer_state, key = update_state
            key, sample_key, shift_key = jax.random.split(key, 3)
            transitions = buffer_sample_fn(buffer_state, sample_key).experience
            shift = jax.random.randint(shift_key, (), 1, m)
            grads, loss_info = jax.grad(loss_fn, has_aux=True)(params.online, params.target, transitions, shift)
            grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="batch")
            grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="device")
            upd, q_opt = q_update_fn(grads, opt_states["q"])
            new_online = optax.apply_updates(params.online, upd)
            n = opt_states["n"] + 1
            sync = (n % target_period) == 0
            new_target = jax.tree_util.tree_map(lambda o, t_: jnp.where(sync, o, t_), new_online, params.target)
            new_opt = {"q": q_opt, "t": opt_states["t"], "n": n}
            return (OnlineAndTarget(new_online, new_target), new_opt, buffer_state, key), loss_info

        update_state, loss_info = jax.lax.scan(_update_epoch, (params, opt_states, buffer_state, key), None,
                                               config.system.epochs)
        params, opt_states, buffer_state, key = update_state
        learner_state = OffPolicyLearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)
        return learner_state, (traj_batch.info, loss_info)

    def learner_fn(learner_state):
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")
        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval)
        return AnakinExperimentOutput(learner_state=learner_state, episode_metrics=episode_info,
                                      train_metrics=loss_info)

    return learner_fn


def learner_setup(env, keys, config):
    n_devices = len(jax.devices())
    A = int(env.action_space().num_values)
    config.system.action_dim = A
    key, net_key = keys
    head = str(config.system.head)
    net = PairNet(num_actions=A, head=head,
                  hidden=512 if head == "rc51" else 128, emb_dim=48 if head == "rc51" else 16,
                  rank=int(config.system.get("pair_rank", 32)), n_atoms=int(config.system.get("rc_atoms", 51)),
                  vmax=float(config.system.get("rc_vmax", 20.0)),
                  additive=bool(config.system.get("all_pairs", False)) and head in ("free", "anti"))
    train_eps, eval_eps = float(config.system.training_epsilon), float(config.system.evaluation_epsilon)
    q_apply = lambda p, o: distrax.EpsilonGreedy(  # noqa: E731
        preferences=net.apply(p, o, method=PairNet.scores), epsilon=train_eps)
    eval_apply = lambda p, o: distrax.EpsilonGreedy(  # noqa: E731
        preferences=net.apply(p, o, method=PairNet.scores), epsilon=eval_eps)

    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = optax.chain(optax.clip_by_global_norm(config.system.max_grad_norm), optax.adam(q_lr, eps=1e-5))
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    online = net.init(net_key, init_x)
    params = OnlineAndTarget(online, online)
    opt_states = {"q": q_optim.init(online),
                  "t": jnp.int32(int(config.system.warmup_steps) * int(config.arch.num_envs)), "n": jnp.int32(0)}

    dummy = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x), action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float), done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False})
    config.system.buffer_size = config.system.total_buffer_size // (n_devices * config.arch.update_batch_size)
    config.system.batch_size = config.system.total_batch_size // (n_devices * config.arch.update_batch_size)
    buffer_fn = fbx.make_item_buffer(max_length=config.system.buffer_size, min_length=config.system.batch_size,
                                     sample_batch_size=config.system.batch_size, add_batches=True,
                                     add_sequences=True)
    buffer_states = buffer_fn.init(dummy)

    learn = get_learner_fn(env, net, q_apply, q_optim.update, (buffer_fn.add, buffer_fn.sample), config)
    learn = jax.pmap(learn, axis_name="device")
    warmup = get_warmup_fn(env, params, q_apply, buffer_fn.add, config)
    warmup = jax.pmap(warmup, axis_name="device")

    key, *env_keys = jax.random.split(key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1)
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(  # noqa: E731
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:])
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(warmup_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])  # noqa: E731
    step_keys, warmup_keys = reshape_keys(jnp.stack(step_keys)), reshape_keys(jnp.stack(warmup_keys))
    replicate_learner = (params, opt_states, buffer_states)
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)  # noqa: E731
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())
    params, opt_states, buffer_states = replicate_learner
    env_states, timesteps, keys, buffer_states = warmup(env_states, timesteps, buffer_states, warmup_keys)
    init_learner_state = OffPolicyLearnerState(params, opt_states, buffer_states, step_keys, env_states, timesteps)
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(online))
    return learn, eval_apply, init_learner_state, n_params


def _mean(x) -> float:
    return float(np.mean(np.asarray(x)))


def run_experiment(_config: DictConfig) -> float:
    config = copy.deepcopy(_config)
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    env, eval_env = environments.make(config=config)
    key, key_e, net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)
    learn, eval_apply, learner_state, n_params = learner_setup(env, (key, net_key), config)
    evaluator, _, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env, key_e=key_e, eval_act_fn=get_distribution_act_fn(config, eval_apply),
        params=learner_state.params.online, config=config)
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (n_devices * config.arch.num_updates_per_eval * config.system.rollout_length
                         * config.arch.update_batch_size * config.arch.num_envs)
    logger = StoixLogger(config)
    os.makedirs(config.system.run_log_dir, exist_ok=True)
    algo = str(config.system.get("algo_name") or f"pair_{config.system.head}_{config.system.term}")
    assert str(config.system.head) in ("anti", "free", "sadqn", "lowrank", "rc51")
    log_path = os.path.join(config.system.run_log_dir, f"{config.env.scenario.task_name}_{algo}_s{config.arch.seed}.jsonl")
    flog = open(log_path, "w")
    flog.write(json.dumps({"config": OmegaConf.to_container(config, resolve=True), "n_params": n_params}) + "\n")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{algo}  params {n_params}  log -> {log_path}{Style.RESET_ALL}")
    t0 = time.time()
    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()
        out = learn(learner_state)
        jax.block_until_ready(out)
        elapsed = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        ep, done_any = get_final_step_metrics(out.episode_metrics)
        rec = {"timestep": t, "env_calls": t, "sps": steps_per_rollout / elapsed}
        if done_any:
            rec["train_return"] = _mean(ep["episode_return"]); rec["train_length"] = _mean(ep["episode_length"])
        rec.update({k: _mean(v) for k, v in out.train_metrics.items()})
        trained_params = unreplicate_batch_dim(out.learner_state.params.online)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)
        ev = evaluator(trained_params, eval_keys)
        jax.block_until_ready(ev)
        rec["eval_return"] = _mean(ev.episode_metrics["episode_return"])
        rec["eval_length"] = _mean(ev.episode_metrics["episode_length"])
        rec["wall"] = time.time() - t0
        logger.log(ev.episode_metrics, t, eval_step, LogEvent.EVAL)
        flog.write(json.dumps(rec) + "\n"); flog.flush()
        learner_state = out.learner_state
    flog.write(json.dumps({"done": True, "wall": time.time() - t0}) + "\n")
    flog.close()
    logger.stop()
    return float(rec["eval_return"])


@hydra.main(config_path="../../configs/default/anakin", config_name="default_ff_rdqn.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    perf = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}pair critic experiment completed{Style.RESET_ALL}")
    return perf


if __name__ == "__main__":
    hydra_entry_point()
