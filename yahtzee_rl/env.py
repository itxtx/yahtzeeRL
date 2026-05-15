"""Batched JAX Yahtzee environment for two-player self-play."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.scoring import score_category, total_score


class EnvState(NamedTuple):
    """Complete batched game state.

    scorecards uses -1 for unfilled categories and concrete scores for filled
    categories. rewards are always from the perspective of the player who acts.
    """

    scorecards: jax.Array
    dice: jax.Array
    rolls_left: jax.Array
    active_player: jax.Array
    done: jax.Array


def _roll_dice(key: jax.Array, batch_size: int) -> jax.Array:
    return jax.random.randint(key, (batch_size, c.NUM_DICE), 1, c.NUM_FACES + 1)


def reset(key: jax.Array, batch_size: int = 1) -> EnvState:
    """Create a batch of fresh games with the first player's opening roll."""
    scorecards = jnp.full(
        (batch_size, c.NUM_PLAYERS, c.NUM_CATEGORIES), -1, dtype=jnp.int32
    )
    return EnvState(
        scorecards=scorecards,
        dice=_roll_dice(key, batch_size),
        rolls_left=jnp.full((batch_size,), c.MAX_ROLLS_LEFT, dtype=jnp.int32),
        active_player=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )


def current_scorecard(state: EnvState) -> jax.Array:
    batch = jnp.arange(state.scorecards.shape[0])
    return state.scorecards[batch, state.active_player]


def observation(state: EnvState) -> dict[str, jax.Array]:
    """Encode observations from the active player's perspective."""
    batch = jnp.arange(state.scorecards.shape[0])
    opponent = 1 - state.active_player
    own = state.scorecards[batch, state.active_player]
    other = state.scorecards[batch, opponent]
    return {
        "own_filled": (own >= 0).astype(jnp.float32),
        "own_scores": jnp.maximum(own, 0).astype(jnp.float32) / 50.0,
        "opp_filled": (other >= 0).astype(jnp.float32),
        "opp_scores": jnp.maximum(other, 0).astype(jnp.float32) / 50.0,
        "dice": state.dice,
        "rolls_left": state.rolls_left,
        "active_player": state.active_player,
    }


def legal_action_mask(state: EnvState) -> jax.Array:
    """Return boolean legal-action masks shaped (batch, 45)."""
    batch_size = state.scorecards.shape[0]
    hold_legal = (state.rolls_left > 0) & (~state.done)
    hold_mask = jnp.broadcast_to(hold_legal[:, None], (batch_size, c.NUM_HOLD_ACTIONS))
    score_mask = (current_scorecard(state) < 0) & (~state.done[:, None])
    mask = jnp.concatenate([hold_mask, score_mask], axis=-1)
    terminal_noop = jnp.zeros_like(mask).at[:, 0].set(state.done)
    return mask | terminal_noop


def action_to_hold_mask(action: jax.Array) -> jax.Array:
    """Convert hold action ids 0..31 to boolean masks over dice positions."""
    bits = jnp.arange(c.NUM_DICE, dtype=jnp.int32)
    return ((action[:, None] >> bits[None, :]) & 1).astype(jnp.bool_)


def _replace_active_score(
    scorecards: jax.Array, active_player: jax.Array, category: jax.Array, score: jax.Array
) -> jax.Array:
    batch = jnp.arange(scorecards.shape[0])
    return scorecards.at[batch, active_player, category].set(score)


def _terminal_reward(scorecards: jax.Array, acting_player: jax.Array) -> jax.Array:
    totals = total_score(scorecards)
    batch = jnp.arange(scorecards.shape[0])
    actor_total = totals[batch, acting_player]
    other_total = totals[batch, 1 - acting_player]
    return jnp.sign(actor_total - other_total).astype(jnp.float32)


def step(state: EnvState, action: jax.Array, key: jax.Array) -> tuple[EnvState, jax.Array]:
    """Advance a batch of games by one action.

    Invalid actions are converted to no-ops with a -1 reward. Policies and MCTS
    should use legal_action_mask, so this is a guard rail rather than gameplay.
    """
    action = action.astype(jnp.int32)
    batch_size = state.scorecards.shape[0]
    legal = legal_action_mask(state)
    chosen_legal = jnp.take_along_axis(legal, action[:, None], axis=-1)[:, 0]

    is_hold = action < c.NUM_HOLD_ACTIONS
    is_score = ~is_hold
    roll_key, next_turn_key = jax.random.split(key)

    hold_mask = action_to_hold_mask(jnp.clip(action, 0, c.NUM_HOLD_ACTIONS - 1))
    rerolled = _roll_dice(roll_key, batch_size)
    dice_after_hold = jnp.where(hold_mask, state.dice, rerolled)
    rolls_left_after_hold = jnp.maximum(state.rolls_left - 1, 0)

    category = jnp.clip(action - c.NUM_HOLD_ACTIONS, 0, c.NUM_CATEGORIES - 1)
    scored_value = score_category(state.dice, category)
    scorecards_after_score = _replace_active_score(
        state.scorecards, state.active_player, category, scored_value
    )
    filled_after_score = scorecards_after_score >= 0
    done_after_score = jnp.all(filled_after_score, axis=(1, 2))
    next_player = 1 - state.active_player
    next_turn_dice = _roll_dice(next_turn_key, batch_size)
    terminal_reward = _terminal_reward(scorecards_after_score, state.active_player)

    terminal_noop = state.done & (action == 0)
    valid_hold = chosen_legal & is_hold & (~state.done)
    valid_score = chosen_legal & is_score & (~state.done)

    next_scorecards = jnp.where(
        valid_score[:, None, None], scorecards_after_score, state.scorecards
    )
    next_dice = jnp.where(
        valid_hold[:, None],
        dice_after_hold,
        jnp.where(valid_score[:, None] & (~done_after_score[:, None]), next_turn_dice, state.dice),
    )
    next_rolls_left = jnp.where(
        valid_hold,
        rolls_left_after_hold,
        jnp.where(valid_score & (~done_after_score), c.MAX_ROLLS_LEFT, state.rolls_left),
    )
    next_active_player = jnp.where(
        valid_score & (~done_after_score), next_player, state.active_player
    )
    next_done = state.done | (valid_score & done_after_score)

    reward = jnp.where(
        valid_score & done_after_score,
        terminal_reward,
        jnp.where(chosen_legal | terminal_noop, 0.0, -1.0),
    )
    next_state = EnvState(
        scorecards=next_scorecards,
        dice=next_dice,
        rolls_left=next_rolls_left.astype(jnp.int32),
        active_player=next_active_player.astype(jnp.int32),
        done=next_done,
    )
    return next_state, reward
