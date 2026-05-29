"""Terminal reward helpers for Yahtzee self-play."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from yahtzee_rl.scoring import total_score

WIN_LOSS = "win_loss"
WIN_LOSS_MARGIN = "win_loss_margin"
REWARD_MODES = (WIN_LOSS, WIN_LOSS_MARGIN)


def terminal_values_from_scores(
    scorecards: jax.Array,
    perspective_players: jax.Array,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
) -> jax.Array:
    """Return terminal values from each requested player's perspective."""
    totals = total_score(scorecards)
    batch = jnp.arange(scorecards.shape[0])
    own_score = totals[batch, perspective_players]
    other_score = totals[batch, 1 - perspective_players]
    margin = (own_score - other_score).astype(jnp.float32)
    win_loss = jnp.sign(margin)

    if reward_mode == WIN_LOSS:
        return win_loss.astype(jnp.float32)
    if reward_mode == WIN_LOSS_MARGIN:
        shaped = win_loss + margin_weight * jnp.tanh(margin / margin_scale)
        return shaped.astype(jnp.float32)
    raise ValueError(f"Unknown reward mode: {reward_mode}")


def terminal_value_from_state(
    scorecards: jax.Array,
    perspective_players: jax.Array,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
) -> jax.Array:
    return terminal_values_from_scores(
        scorecards,
        perspective_players,
        reward_mode=reward_mode,
        margin_weight=margin_weight,
        margin_scale=margin_scale,
    )
