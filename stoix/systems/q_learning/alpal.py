"""Advantage-learning (AL) / persistent AL (PAL) penalties for Relative-Q (2026-11-03).
Bellemare et al. 2016, sample forms with the correction from the TARGET network:
  AL : target = r + g(1-d) max_b Q-(s',b) - alpha * P_AL,   P_AL  = G-(s,a)
  PAL: target = r + g(1-d) max_b Q-(s',b) - alpha * P_PAL,  P_PAL = min{ G-(s,a), (1-d) G-(s',a) }
  G-(s,a) = max_b v-(s,b) - v-(s,a): same-state gap of the greedy read-out v- in Q units (invariant to a constant
  shift of Q, so the RDQN scalar c does not enter).  (PAL = max of the two sampled targets = smallest penalty.)
CW-Pair: per-endpoint penalties enter the cross-weighted residual as  xi' = xi - alpha (v_y P(x) - v_x P(y)).
"""
import jax
import jax.numpy as jnp


def al_penalty(v_cur, v_next, act, done, mode: str):
    """v_cur, v_next: [B, A] target-network read-outs at s and s' (Q units); act [B]; done [B] -> penalty [B] (>= 0)."""
    ar = jnp.arange(act.shape[0])
    G = jnp.max(v_cur, -1) - v_cur[ar, act]
    if mode == "al":
        return jax.lax.stop_gradient(G)
    assert mode == "pal", mode
    Gn = jnp.max(v_next, -1) - v_next[ar, act]
    return jax.lax.stop_gradient(jnp.minimum(G, (1.0 - done.astype(jnp.float32)) * Gn))
