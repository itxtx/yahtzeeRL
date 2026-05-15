"""Self-play rollout generation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import yahtzee_rl.constants as c
from yahtzee_rl.env import EnvState, observation, reset, step
from yahtzee_rl.mcts import search_policy
from yahtzee_rl.scoring import total_score

MAX_GAME_STEPS = c.NUM_PLAYERS * c.NUM_CATEGORIES * (c.MAX_ROLLS_LEFT + 1)


class Trajectory(NamedTuple):
    own_filled: jax.Array
    own_scores: jax.Array
    opp_filled: jax.Array
    opp_scores: jax.Array
    dice: jax.Array
    rolls_left: jax.Array
    active_player: jax.Array
    action_weights: jax.Array
    valid: jax.Array
    returns: jax.Array


def _winner_values(state: EnvState, acting_players: jax.Array) -> jax.Array:
    totals = total_score(state.scorecards)
    player0 = totals[:, 0]
    player1 = totals[:, 1]
    result_for_player0 = jnp.sign(player0 - player1).astype(jnp.float32)
    return jnp.where(acting_players == 0, result_for_player0, -result_for_player0)


def generate_self_play(
    model,
    params,
    rng_key: jax.Array,
    batch_size: int,
    num_simulations: int,
) -> tuple[Trajectory, dict[str, jax.Array]]:
    """Generate a fixed-length batch of two-player self-play games."""
    reset_key, scan_key = jax.random.split(rng_key)
    state = reset(reset_key, batch_size)

    def scan_step(carry, _):
        current_state, key = carry
        key, search_key, env_key = jax.random.split(key, 3)
        obs = observation(current_state)
        policy = search_policy(
            model, params, current_state, search_key, num_simulations=num_simulations
        )
        action = policy.action.astype(jnp.int32)
        next_state, _ = step(current_state, action, env_key)
        valid = ~current_state.done
        frame = (
            obs["own_filled"],
            obs["own_scores"],
            obs["opp_filled"],
            obs["opp_scores"],
            obs["dice"],
            obs["rolls_left"],
            obs["active_player"],
            policy.action_weights,
            valid,
        )
        return (next_state, key), frame

    (final_state, _), frames = jax.lax.scan(
        scan_step, (state, scan_key), None, length=MAX_GAME_STEPS
    )
    acting_players = frames[6]
    valid = frames[8]
    returns = jnp.where(valid, _winner_values(final_state, acting_players), 0.0)

    trajectory = Trajectory(
        own_filled=frames[0],
        own_scores=frames[1],
        opp_filled=frames[2],
        opp_scores=frames[3],
        dice=frames[4],
        rolls_left=frames[5],
        active_player=frames[6],
        action_weights=frames[7],
        valid=valid,
        returns=returns,
    )
    metrics = {
        "mean_player0_score": jnp.mean(total_score(final_state.scorecards)[:, 0]),
        "mean_player1_score": jnp.mean(total_score(final_state.scorecards)[:, 1]),
        "draw_rate": jnp.mean(total_score(final_state.scorecards)[:, 0] == total_score(final_state.scorecards)[:, 1]),
    }
    return trajectory, metrics


def trajectory_observation(trajectory: Trajectory) -> dict[str, jax.Array]:
    """Flatten time and batch dimensions into a model observation."""
    time_steps, batch_size = trajectory.rolls_left.shape

    def flat(x):
        return x.reshape((time_steps * batch_size,) + x.shape[2:])

    return {
        "own_filled": flat(trajectory.own_filled),
        "own_scores": flat(trajectory.own_scores),
        "opp_filled": flat(trajectory.opp_filled),
        "opp_scores": flat(trajectory.opp_scores),
        "dice": flat(trajectory.dice),
        "rolls_left": trajectory.rolls_left.reshape((-1,)),
        "active_player": trajectory.active_player.reshape((-1,)),
    }
