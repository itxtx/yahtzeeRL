"""MCTS policy wrapper using DeepMind mctx."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mctx

from yahtzee_rl.env import EnvState, legal_action_mask, observation, step
from yahtzee_rl.model import masked_logits
from yahtzee_rl.rewards import WIN_LOSS_MARGIN, terminal_value_from_state


def _predict(model, params, state: EnvState) -> tuple[jax.Array, jax.Array]:
    obs = observation(state)
    logits, value = model.apply(params, obs)
    return masked_logits(logits, legal_action_mask(state)), value


def root_output(model, params, state: EnvState) -> mctx.RootFnOutput:
    logits, value = _predict(model, params, state)
    return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)


def make_recurrent_fn(
    model,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
):
    """Build an mctx recurrent_fn around the real Yahtzee transition."""

    def recurrent_fn(params, rng_key, action, embedding: EnvState):
        next_state, env_reward = step(embedding, action, rng_key)
        logits, value = _predict(model, params, next_state)
        player_changed = next_state.active_player != embedding.active_player
        discount = jnp.where(player_changed, -1.0, 1.0)
        discount = jnp.where(next_state.done, 0.0, discount).astype(jnp.float32)
        terminal_reward = terminal_value_from_state(
            next_state.scorecards,
            embedding.active_player,
            reward_mode=reward_mode,
            margin_weight=margin_weight,
            margin_scale=margin_scale,
        )
        reward = jnp.where(next_state.done, terminal_reward, env_reward)
        out = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return out, next_state

    return recurrent_fn


def search_policy(
    model,
    params,
    state: EnvState,
    rng_key: jax.Array,
    num_simulations: int = 32,
    temperature: float = 1.0,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
) -> mctx.PolicyOutput:
    """Run batched Gumbel MuZero search and return the mctx policy output."""
    root = root_output(model, params, state)
    return mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=make_recurrent_fn(
            model,
            reward_mode=reward_mode,
            margin_weight=margin_weight,
            margin_scale=margin_scale,
        ),
        num_simulations=num_simulations,
        invalid_actions=~legal_action_mask(state),
        gumbel_scale=temperature,
    )
