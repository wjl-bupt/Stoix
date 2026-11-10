"""R-PPO (2026-11-01): PPO whose critic is split into a network part and a global scalar,
    V_hat(s) = v_theta(s) + c.
Rollout values, bootstrap values and (standard) GAE all use V_hat.  Critic update per minibatch, with
G = GAE lambda-return targets (fixed during the epochs) and residual u = G - c - v_theta(s):
  * network:  regress v_theta on  y = G - c - sg(mean u)   (=> gradient of 0.5*mean(u - mean u)^2, i.e. the
              centred / gauge-free loss; the usual PPO value clipping is applied to the network part, with
              v_old = v_theta at rollout time);  Adam + global-norm clipping exactly as in ff_ppo.
  * scalar :  c <- c + c_lr * mean(u)   (plain SGD on 0.5*mean(u)^2, not clipped, not decayed; c_lr=1 => the
              minibatch mean residual is absorbed by c in one step).
system.rppo=False gives plain PPO (c is frozen at 0 and the network regresses on G) and is checked to reproduce
ff_ppo.
system.adv_estimator=rgae (2026-11-01): the ACTOR uses RVL's R-GAE instead of GAE (critic unchanged, still trained
on the standard-GAE lambda-returns).  R-GAE as in the official RVL code (rv/loss/gae.py, value_gae_init='zeros',
i.e. no trajectory ranking): relative values are accumulated from 0 at the start of every rollout and every new
episode; for a consistent critic this equals GAE with V_hat(s) - V_hat(s_ref), s_ref = first state of the episode
segment inside the rollout (c cancels).  Advantages are then standardised as usual.  Per-evaluation records go to <system.run_log_dir>/<task>_<algo>_s<seed>.jsonl.
"""
import copy
import json
import os
import time
from typing import Any, Dict, NamedTuple, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, get_final_step_metrics

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    AnakinExperimentOutput,
    CriticApply,
    LearnerFn,
    OnPolicyLearnerState,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.base import FeedForwardCritic as Critic
from stoix.systems.ppo.ppo_types import PPOTransition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.running_statistics import (
    create_with_running_statistics,
    initialize_statistics_from_data,
    normalize,
    update_statistics,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
import numpy as np


class RPPOTransition(NamedTuple):
    done: chex.Array
    truncated: chex.Array
    action: chex.Array
    value: chex.Array        # V_hat = v_theta + c at rollout time
    value_net: chex.Array    # v_theta at rollout time (for value clipping of the network part)
    reward: chex.Array
    bootstrap_value: chex.Array
    log_prob: chex.Array
    obs: chex.Array
    info: Dict


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn, c_update_fn = update_fns
    rppo = bool(config.system.rppo)

    def v_net_fn(critic_params, obs):
        return critic_apply_fn(critic_params["net"], obs)

    def v_hat_fn(critic_params, obs):
        return v_net_fn(critic_params, obs) + critic_params["c"]

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: OnPolicyLearnerState, _: Any
        ) -> Tuple[OnPolicyLearnerState, PPOTransition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep = learner_state

            # GET OBSERVATION
            observation = last_timestep.observation

            # Get running statistics if normalizing observations.
            # It would not be present if observation normalization is not implemented/setup.
            running_statistics = getattr(learner_state, "running_statistics", None)
            if running_statistics is not None:
                observation = normalize(observation, running_statistics)

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params, observation)
            value_net = v_net_fn(params.critic_params, observation)
            value = value_net + params.critic_params["c"]
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]
            # Save bootstrap value for the next step.
            # Due to auto-resetting and truncation, we have to specifically save the bootstrap value
            # for the next (potentially final) observation.
            bootstrap_obs = timestep.extras["next_obs"]
            if running_statistics is not None:
                bootstrap_obs = normalize(bootstrap_obs, running_statistics)
            bootstrap_value = v_hat_fn(params.critic_params, bootstrap_obs)

            transition = RPPOTransition(
                done,
                truncated,
                action,
                value,
                value_net,
                timestep.reward,
                bootstrap_value,
                log_prob,
                last_timestep.observation,
                info,
            )
            # Replace the learner state with the new environment state and timestep.
            learner_state = learner_state._replace(
                key=key,
                env_state=env_state,
                timestep=timestep,
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # UNPACK LEARNER STATE
        params, opt_states, key, _, _ = learner_state

        # IF NORMALIZING OBSERVATIONS, UPDATE RUNNING STATISTICS
        running_statistics = getattr(learner_state, "running_statistics", None)
        if running_statistics is not None:
            # We get the raw observations from the trajectory batch.
            raw_obs = traj_batch.obs
            # We normalize with pre-updated statistics
            normalized_obs = normalize(traj_batch.obs, running_statistics)
            # Update running statistics
            running_statistics = update_statistics(
                running_statistics,
                raw_obs,
                pmap_axes=["device", "batch"],
                std_min_value=5e-4,
                std_max_value=5e4,
            )
            learner_state = learner_state._replace(running_statistics=running_statistics)  # type: ignore
            # We then replace the observations in the trajectory batch with the normalized ones.
            traj_batch = traj_batch._replace(obs=normalized_obs)

        # CALCULATE ADVANTAGE
        v_tm1 = traj_batch.value
        r_t = traj_batch.reward * config.system.reward_scale
        v_t = traj_batch.bootstrap_value
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            v_tm1=v_tm1,
            v_t=v_t,
            time_major=True,
            standardize_advantages=False,
            truncation_t=traj_batch.truncated,
        )
        gae_raw = advantages
        if str(config.system.adv_estimator) == "rgae":
            # reference value = V_hat at the first state of the current episode segment within this rollout
            ends = jnp.logical_or(traj_batch.done, traj_batch.truncated)  # (T, envs): step t ended the episode

            def _ref_body(ref, xs):
                v_cur, prev_end = xs
                ref = jnp.where(prev_end, v_cur, ref)
                return ref, ref

            prev_end = jnp.concatenate([jnp.ones_like(ends[:1]), ends[:-1]], axis=0)
            _, ref = jax.lax.scan(_ref_body, v_tm1[0], (v_tm1, prev_end))
            advantages, _ = batch_truncated_generalized_advantage_estimation(
                r_t,
                d_t,
                config.system.gae_lambda,
                v_tm1=v_tm1 - ref,
                v_t=v_t - ref,
                time_major=True,
                standardize_advantages=False,
                truncation_t=traj_batch.truncated,
            )
        adv_diff = advantages - gae_raw  # = B_t (0 for standard GAE)
        if config.system.standardize_advantages:
            advantages = jax.nn.standardize(advantages, axis=(0, 1))

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: RPPOTransition,
                    gae: chex.Array,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # RERUN NETWORK
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # CALCULATE ACTOR LOSS
                    loss_actor = ppo_clip_loss(
                        log_prob, traj_batch.log_prob, gae, config.system.clip_eps
                    )
                    entropy = actor_policy.entropy().mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    loss_info = {
                        "actor_loss": loss_actor,
                        "entropy": entropy,
                        "advantages": gae,
                    }
                    return total_loss_actor, loss_info

                def _critic_loss_fn(
                    net_params: FrozenDict,
                    c: chex.Array,
                    traj_batch: RPPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Network part of the critic loss (c enters only as a constant)."""
                    value_net = critic_apply_fn(net_params, traj_batch.obs)
                    c_sg = jax.lax.stop_gradient(c)
                    resid = jax.lax.stop_gradient(targets - c_sg - value_net)  # u = G - c - v_theta
                    resid_mean = resid.mean()
                    if rppo:
                        y = targets - c_sg - resid_mean   # centred target: grad = -(u - mean u) grad v
                    else:
                        y = targets - c_sg                # c == 0 in plain PPO -> y = G
                    value_loss = clipped_value_loss(
                        value_net, traj_batch.value_net, y, config.system.clip_eps
                    )
                    critic_total_loss = config.system.vf_coef * value_loss
                    loss_info = {
                        "value_loss": value_loss,
                        "pred_value": value_net + c_sg,
                        "v_net_mean": value_net.mean(),
                        "target_value": targets,
                        "resid_mean": resid_mean,
                        "resid_std": resid.std(),
                        "c": c_sg,
                    }
                    return critic_total_loss, loss_info

                def _c_loss_fn(c: chex.Array, net_params: FrozenDict, traj_batch, targets) -> chex.Array:
                    value_net = jax.lax.stop_gradient(critic_apply_fn(net_params, traj_batch.obs))
                    m = (targets - c - value_net).mean()
                    return 0.5 * jnp.square(m)   # SGD(c_lr) => c <- c + c_lr * mean(u)

                # CALCULATE ACTOR LOSS
                actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
                actor_grads, actor_loss_info = actor_grad_fn(
                    params.actor_params, traj_batch, advantages
                )

                # CALCULATE CRITIC LOSS
                critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
                critic_grads, critic_loss_info = critic_grad_fn(
                    params.critic_params["net"], params.critic_params["c"], traj_batch, targets
                )
                c_grad = jax.grad(_c_loss_fn)(
                    params.critic_params["c"], params.critic_params["net"], traj_batch, targets
                )
                if not rppo:
                    c_grad = jnp.zeros_like(c_grad)

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info, critic_grads, critic_loss_info, c_grad = jax.lax.pmean(
                    (actor_grads, actor_loss_info, critic_grads, critic_loss_info, c_grad),
                    axis_name="batch",
                )
                # pmean over devices.
                actor_grads, actor_loss_info, critic_grads, critic_loss_info, c_grad = jax.lax.pmean(
                    (actor_grads, actor_loss_info, critic_grads, critic_loss_info, c_grad),
                    axis_name="device",
                )

                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                net_opt_state, c_opt_state = opt_states.critic_opt_state
                critic_updates, net_new_opt_state = critic_update_fn(critic_grads, net_opt_state)
                net_new_params = optax.apply_updates(params.critic_params["net"], critic_updates)
                c_updates, c_new_opt_state = c_update_fn(c_grad, c_opt_state)
                c_new = optax.apply_updates(params.critic_params["c"], c_updates)
                critic_new_params = {"net": net_new_params, "c": c_new}
                critic_new_opt_state = (net_new_opt_state, c_new_opt_state)

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = ActorCriticParams(actor_new_params, critic_new_params)
                new_opt_state = ActorCriticOptStates(actor_new_opt_state, critic_new_opt_state)

                # PACK LOSS INFO
                loss_info = {
                    **actor_loss_info,
                    **critic_loss_info,
                }
                loss_info = jax.tree_util.tree_map(jnp.mean, loss_info)
                return (new_params, new_opt_state), loss_info

            (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            ) = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = learner_state._replace(params=params, opt_states=opt_states, key=key)
        metric = traj_batch.info
        loss_info["rollout_reward_per_step"] = traj_batch.reward.mean() * jnp.ones(())
        loss_info["term_frac"] = traj_batch.done.astype(jnp.float32).mean() * jnp.ones(())
        loss_info["gae_target_mean"] = targets.mean() * jnp.ones(())
        loss_info["v_hat_rollout_mean"] = traj_batch.value.mean() * jnp.ones(())
        loss_info["c_end"] = params.critic_params["c"] * jnp.ones(())
        loss_info["adv_raw_std"] = advantages.std() * 0 + gae_raw.std()
        loss_info["rgae_minus_gae_mean"] = adv_diff.mean()
        loss_info["rgae_minus_gae_std"] = adv_diff.std()
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (ActorCriticParams): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (WrapperState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def _collect_obs_norm_rollouts(
    env: Environment,
    key: chex.PRNGKey,
    config: DictConfig,
) -> chex.ArrayTree:
    """Collect observations for observation normalization.
    This function collects observations from the environment by taking random actions
    for a specified number of warmup steps."""
    num_warmup_obs = config.system.obs_norm_warmup_steps * config.arch.total_num_envs
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Initializing observation normalization with "
        f"{num_warmup_obs} observations..."
        f"Be aware, we do not count this in the timestep budget.{Style.RESET_ALL}"
    )

    @jax.jit
    def _warmup_step(env_state: Any, step_key: chex.PRNGKey) -> Tuple[Any, chex.ArrayTree]:
        """Single warmup step: random action -> environment step -> collect observation."""
        action = jax.random.randint(
            step_key, (config.arch.total_num_envs,), 0, config.system.action_dim
        )
        env_state, timestep = env.step(env_state, action)
        return env_state, timestep.observation

    # Initialize environments for data collection
    key, *env_keys = jax.random.split(key, config.arch.total_num_envs + 1)
    env_states, initial_timesteps = env.reset(jnp.stack(env_keys))

    # Collect warmup observations through random actions
    step_keys = jax.random.split(key, config.system.obs_norm_warmup_steps)
    _, warmup_observations = jax.lax.scan(_warmup_step, env_states, step_keys)

    # Combine initial reset observations with warmup observations
    all_observations = jax.tree.map(
        lambda reset_obs, warmup_obs: jnp.concatenate(
            [
                reset_obs[
                    jnp.newaxis,
                ],
                warmup_obs,
            ],
            axis=0,
        ),
        initial_timesteps.observation,
        warmup_observations,
    )
    return all_observations


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[OnPolicyLearnerState], Actor, OnPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso, critic_head=critic_head)

    actor_lr = make_learning_rate(
        config.system.actor_lr, config, config.system.epochs, config.system.num_minibatches
    )
    critic_lr = make_learning_rate(
        config.system.critic_lr, config, config.system.epochs, config.system.num_minibatches
    )

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )
    c_optim = optax.sgd(float(config.system.c_lr))

    # Initialise observation
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_net_params = critic_network.init(critic_net_key, init_x)
    c0 = jnp.asarray(float(config.system.c_init), dtype=jnp.float32)
    critic_params = {"net": critic_net_params, "c": c0}
    critic_opt_state = (critic_optim.init(critic_net_params), c_optim.init(c0))

    # Pack params.
    params = ActorCriticParams(actor_params, critic_params)

    actor_network_apply_fn = actor_network.apply
    critic_network_apply_fn = critic_network.apply

    # Pack apply and update functions.
    apply_fns = (actor_network_apply_fn, critic_network_apply_fn)
    update_fns = (actor_optim.update, critic_optim.update, c_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
    )

    # If normalizing observations, initialize running statistics.
    # This will add running statistics to the learner state.
    if config.system.normalize_observations:
        dummy_obs = jax.tree.map(lambda x: x.squeeze(0), init_x)
        warmup_observations = _collect_obs_norm_rollouts(env, key, config)
        running_statistics = initialize_statistics_from_data(dummy_obs, warmup_observations)
        running_statistics = jax.tree.map(broadcast, running_statistics)
        running_statistics = flax.jax_utils.replicate(running_statistics, devices=jax.devices())
        init_learner_state = create_with_running_statistics(
            state=init_learner_state, running_statistics=running_statistics
        )

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, actor_network.apply),
        params=learner_state.params.actor_params,
        config=config,
    )

    # Calculate environment steps per evaluation.
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Per-evaluation jsonl log (same convention as ff_rdqn.py).
    os.makedirs(config.system.run_log_dir, exist_ok=True)
    algo = str(config.system.get("algo_name") or ("rppo" if config.system.rppo else "ppo"))
    log_path = os.path.join(config.system.run_log_dir,
                            f"{config.env.scenario.task_name}_{algo}_s{config.arch.seed}.jsonl")
    flog = open(log_path, "w")
    flog.write(json.dumps({"config": OmegaConf.to_container(config, resolve=True)}) + "\n")
    print(f"{Fore.YELLOW}{Style.BRIGHT}{algo}  log -> {log_path}{Style.RESET_ALL}")
    t_start = time.time()

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_learner_state = unreplicate_batch_dim(learner_state)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        rec = {"timestep": t, "sps": steps_per_rollout / elapsed_time}
        if ep_completed:
            rec["train_return"] = float(np.mean(np.asarray(episode_metrics["episode_return"])))
            rec["train_length"] = float(np.mean(np.asarray(episode_metrics["episode_length"])))
        for k_, v_ in train_metrics.items():
            rec[k_] = float(np.mean(np.asarray(v_)))
        rec["c_end"] = float(np.asarray(train_metrics["c_end"]).reshape(-1)[-1])
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (
            config.system.epochs * config.system.num_minibatches
        )
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.actor_params
        )  # Select only actor params
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        running_statistics = getattr(learner_output.learner_state, "running_statistics", None)
        if running_statistics is not None:
            running_statistics = unreplicate_batch_dim(running_statistics)
        evaluator_output = evaluator(trained_params, eval_keys, running_statistics)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)
        rec["eval_return"] = float(episode_return)
        rec["eval_length"] = float(np.mean(np.asarray(evaluator_output.episode_metrics["episode_length"])))
        rec["wall"] = time.time() - t_start
        flog.write(json.dumps(rec) + "\n")
        flog.flush()

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_learner_state = copy.deepcopy(unreplicate_batch_dim(learner_state))
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        best_params = best_learner_state.params.actor_params
        best_running_statistics = getattr(best_learner_state, "running_statistics", None)
        evaluator_output = absolute_metric_evaluator(
            best_params, eval_keys, best_running_statistics
        )
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    flog.write(json.dumps({"done": True, "wall": time.time() - t_start}) + "\n")
    flog.close()
    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance


@hydra.main(
    config_path="../../../configs/default/anakin",
    config_name="default_ff_rppo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    t0 = time.time()
    eval_performance = run_experiment(cfg)

    print(
        f"{Fore.CYAN}{Style.BRIGHT}R-PPO/PPO experiment completed in "
        f"{time.time() - t0:.2f} seconds.{Style.RESET_ALL}"
    )
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
