"""Self-play rollout generation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import yahtzee_rl.constants as c
from yahtzee_rl.env import EnvState, observation, reset, step
from yahtzee_rl.mcts import search_policy
from yahtzee_rl.rewards import WIN_LOSS_MARGIN, terminal_values_from_scores
from yahtzee_rl.scoring import total_score

MAX_GAME_STEPS = c.NUM_PLAYERS * c.NUM_CATEGORIES * (c.MAX_ROLLS_LEFT + 1)


class Trajectory(NamedTuple):
    own_filled: jax.Array
    own_scores: jax.Array
    opp_filled: jax.Array
    opp_scores: jax.Array
    upper_own: jax.Array
    upper_opp: jax.Array
    dice_counts: jax.Array
    rolls_left: jax.Array
    active_player: jax.Array
    action_weights: jax.Array
    search_values: jax.Array
    valid: jax.Array
    returns: jax.Array


def terminal_values(
    state: EnvState,
    acting_players: jax.Array,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
) -> jax.Array:
    flat_players = acting_players.reshape((-1,))
    repeated_scorecards = jnp.repeat(
        state.scorecards[None, :, :, :], acting_players.shape[0], axis=0
    ).reshape((-1,) + state.scorecards.shape[1:])
    values = terminal_values_from_scores(
        repeated_scorecards,
        flat_players,
        reward_mode=reward_mode,
        margin_weight=margin_weight,
        margin_scale=margin_scale,
    )
    return values.reshape(acting_players.shape)


def generate_self_play(
    model,
    params,
    rng_key: jax.Array,
    batch_size: int,
    num_simulations: int,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
) -> tuple[Trajectory, dict[str, jax.Array]]:
    """Generate a fixed-length batch of two-player self-play games."""
    reset_key, scan_key = jax.random.split(rng_key)
    state = reset(reset_key, batch_size)

    def scan_step(carry, _):
        current_state, key = carry
        key, search_key, env_key = jax.random.split(key, 3)
        obs = observation(current_state)
        policy = search_policy(
            model,
            params,
            current_state,
            search_key,
            num_simulations=num_simulations,
            reward_mode=reward_mode,
            margin_weight=margin_weight,
            margin_scale=margin_scale,
        )
        action = policy.action.astype(jnp.int32)
        # Root value of the (decision-masked) search tree: an improved value
        # estimate used to reduce the variance of the Monte Carlo target.
        search_value = policy.search_tree.summary().value
        next_state, _ = step(current_state, action, env_key)
        valid = ~current_state.done
        frame = (
            obs["own_filled"],
            obs["own_scores"],
            obs["opp_filled"],
            obs["opp_scores"],
            obs["upper_own"],
            obs["upper_opp"],
            obs["dice_counts"],
            obs["rolls_left"],
            obs["active_player"],
            policy.action_weights,
            search_value,
            valid,
        )
        return (next_state, key), frame

    (final_state, _), frames = jax.lax.scan(
        scan_step, (state, scan_key), None, length=MAX_GAME_STEPS
    )
    acting_players = frames[8]
    valid = frames[11]
    shaped_values = terminal_values(
        final_state,
        acting_players,
        reward_mode=reward_mode,
        margin_weight=margin_weight,
        margin_scale=margin_scale,
    )
    returns = jnp.where(valid, shaped_values, 0.0)
    totals = total_score(final_state.scorecards)
    score_margin_player0 = totals[:, 0] - totals[:, 1]

    trajectory = Trajectory(
        own_filled=frames[0],
        own_scores=frames[1],
        opp_filled=frames[2],
        opp_scores=frames[3],
        upper_own=frames[4],
        upper_opp=frames[5],
        dice_counts=frames[6],
        rolls_left=frames[7],
        active_player=frames[8],
        action_weights=frames[9],
        search_values=frames[10],
        valid=valid,
        returns=returns,
    )
    metrics = {
        "mean_player0_score": jnp.mean(totals[:, 0]),
        "mean_player1_score": jnp.mean(totals[:, 1]),
        "player0_win_rate": jnp.mean(totals[:, 0] > totals[:, 1]),
        "player1_win_rate": jnp.mean(totals[:, 1] > totals[:, 0]),
        "draw_rate": jnp.mean(totals[:, 0] == totals[:, 1]),
        "mean_score_margin_player0": jnp.mean(score_margin_player0),
        "mean_abs_score_margin": jnp.mean(jnp.abs(score_margin_player0)),
        "mean_shaped_return": jnp.sum(jnp.where(valid, returns, 0.0))
        / jnp.maximum(jnp.sum(valid), 1),
    }
    return trajectory, metrics


def trajectory_observation(trajectory: Trajectory) -> dict[str, jax.Array]:
    """Flatten time and batch dimensions into a model observation."""
    time_steps, batch_size = trajectory.rolls_left.shape

    def flat(x):
        return x.reshape((time_steps * batch_size,) + x.shape[2:])

    flat_rolls = trajectory.rolls_left.reshape((-1,))
    zeros = jnp.zeros_like(flat_rolls)
    return {
        "own_filled": flat(trajectory.own_filled),
        "own_scores": flat(trajectory.own_scores),
        "opp_filled": flat(trajectory.opp_filled),
        "opp_scores": flat(trajectory.opp_scores),
        "upper_own": trajectory.upper_own.reshape((-1,)),
        "upper_opp": trajectory.upper_opp.reshape((-1,)),
        "dice_counts": flat(trajectory.dice_counts),
        "num_unknown": zeros,
        "rolls_left": flat_rolls,
        "opponent_to_move": zeros.astype(jnp.float32),
        "active_player": trajectory.active_player.reshape((-1,)),
    }
