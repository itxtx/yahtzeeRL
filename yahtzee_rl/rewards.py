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
    if not 0.0 <= margin_weight <= 1.0:
        raise ValueError(
            f"margin_weight must be in [0, 1] to keep the shaped reward within "
            f"[-1, 1]; got {margin_weight}"
        )
    totals = total_score(scorecards)
    batch = jnp.arange(scorecards.shape[0])
    own_score = totals[batch, perspective_players]
    other_score = totals[batch, 1 - perspective_players]
    margin = (own_score - other_score).astype(jnp.float32)
    win_loss = jnp.sign(margin)

    if reward_mode == WIN_LOSS:
        return win_loss.astype(jnp.float32)
    if reward_mode == WIN_LOSS_MARGIN:
        # Keep the shaped target within [-1, 1] so it matches the tanh-bounded
        # value head. margin_weight splits the unit reward between a win/loss
        # baseline (1 - margin_weight) and a margin-sensitive term (margin_weight).
        shaped = win_loss * (1.0 - margin_weight) + margin_weight * jnp.tanh(
            margin / margin_scale
        )
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
