"""2026-11-08: Kinetix symbolic-entity transformer encoder for off-policy systems (cw_continuous.py).
Same structure as the official Kinetix ActorCriticTransformer embedding (kinetix/models/transformer_model.py):
per-entity tanh Dense embeddings (+ shape-type id), gated transformer over shapes with joint/thruster message
passing (kinetix Transformer), scene embedding.  Official hyperparameters (Kinetix paper Table 7): encoder size 128,
transformer (qkv) size 16, 8 heads, 2 layers.  aggregate_mode = dummy_and_mean (our choice; the paper does not state it).
Input: EntityObservation (or dict) with leading batch dim [B, ...]; output [B, encoder_size]."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from kinetix.models.transformer_model import Transformer
from kinetix.render.renderer_symbolic_entity import EntityObservation


class KinetixTransformerEncoder(nn.Module):
    encoder_size: int = 128
    transformer_size: int = 16
    num_heads: int = 8
    depth: int = 2
    aggregate_mode: str = "dummy_and_mean"

    @nn.compact
    def __call__(self, obs):
        if isinstance(obs, dict):
            obs = EntityObservation(**obs)
        obs = jax.tree_util.tree_map(lambda x: x[None], obs)          # add T = 1 (official code is [T, B, ...])
        act = nn.tanh

        def enc(features, entity_id, concat=True):
            k = 1 if concat else 0
            e = act(nn.Dense(self.encoder_size - k, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(features))
            if concat:
                return jnp.concatenate([e, jnp.full((*e.shape[:3], 1), float(entity_id))], axis=-1)
            return e

        circle = enc(obs.circles, 0)
        polygon = enc(obs.polygons, 1)
        joint = enc(obs.joints, -1, False)
        thruster = enc(obs.thrusters, -1, False)
        shapes = jnp.concatenate([polygon, circle], axis=2)
        shape_mask = jnp.concatenate([obs.polygon_mask, obs.circle_mask], axis=2)

        def mask_out(flat_active, m):
            return m & flat_active[:, None] & flat_active[None, :]

        joint_idx, thr_idx = obs.joint_indexes, obs.thruster_indexes
        if self.aggregate_mode in ("dummy", "dummy_and_mean"):
            T, B, _, K = circle.shape
            shapes = jnp.concatenate([jnp.ones((T, B, 1, K)), shapes], axis=2)
            shape_mask = jnp.concatenate([jnp.ones((T, B, 1), dtype=bool), shape_mask], axis=2)
            N = obs.attention_mask.shape[-1]
            full = jnp.ones((T, B, obs.attention_mask.shape[2], N + 1, N + 1), dtype=bool).at[:, :, :, 1:, 1:].set(
                obs.attention_mask)
            mask = jax.vmap(jax.vmap(mask_out))(shape_mask, full)
            joint_idx = joint_idx + 1
            thr_idx = thr_idx + 1
        else:
            mask = obs.attention_mask
        emb = Transformer(num_layers=self.depth, num_heads=self.num_heads, qkv_features=self.transformer_size,
                          encoder_size=self.encoder_size, gating=True, gating_bias=0.0)(
            shapes, jnp.repeat(mask, repeats=self.num_heads // mask.shape[2], axis=2),
            joint, obs.joint_mask, joint_idx, thruster, obs.thruster_mask, thr_idx)
        if self.aggregate_mode in ("mean", "dummy_and_mean"):
            emb = jnp.mean(emb, axis=2, where=shape_mask[..., None])
        else:
            emb = emb[:, :, 0]
        return emb[0]
