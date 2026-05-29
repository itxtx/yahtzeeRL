"""JAX scoring helpers for simplified standard Yahtzee."""

import jax
import jax.numpy as jnp

from yahtzee_rl import constants as c


def dice_counts(dice: jax.Array) -> jax.Array:
    """Return counts for faces 1..6 for dice shaped (..., 5)."""
    one_hot = jax.nn.one_hot(dice - 1, c.NUM_FACES, dtype=jnp.int32)
    return jnp.sum(one_hot, axis=-2)


def score_categories(dice: jax.Array) -> jax.Array:
    """Score all 13 categories for a batch of dice.

    Args:
        dice: Integer array with shape (..., 5) and values in 1..6.

    Returns:
        Integer array with shape (..., 13).
    """
    counts = dice_counts(dice)
    faces = jnp.arange(1, c.NUM_FACES + 1, dtype=jnp.int32)
    total = jnp.sum(dice, axis=-1)

    upper = counts * faces
    has_three = jnp.any(counts >= 3, axis=-1)
    has_four = jnp.any(counts >= 4, axis=-1)
    has_five = jnp.any(counts == 5, axis=-1)
    has_pair = jnp.any(counts == 2, axis=-1)
    has_triple = jnp.any(counts == 3, axis=-1)

    present = counts > 0
    small_straight = (
        jnp.all(present[..., 0:4], axis=-1)
        | jnp.all(present[..., 1:5], axis=-1)
        | jnp.all(present[..., 2:6], axis=-1)
    )
    large_straight = jnp.all(present[..., 0:5], axis=-1) | jnp.all(
        present[..., 1:6], axis=-1
    )

    lower = jnp.stack(
        [
            jnp.where(has_three, total, 0),
            jnp.where(has_four, total, 0),
            jnp.where(has_triple & has_pair, 25, 0),
            jnp.where(small_straight, 30, 0),
            jnp.where(large_straight, 40, 0),
            jnp.where(has_five, 50, 0),
            total,
        ],
        axis=-1,
    ).astype(jnp.int32)

    return jnp.concatenate([upper, lower], axis=-1)


def score_category(dice: jax.Array, category: jax.Array) -> jax.Array:
    """Score selected category indices for dice shaped (..., 5)."""
    scores = score_categories(dice)
    return jnp.take_along_axis(scores, category[..., None], axis=-1)[..., 0]


def total_score(scorecard: jax.Array) -> jax.Array:
    """Return total score for scorecards shaped (..., 13), where -1 is empty."""
    filled = jnp.maximum(scorecard, 0)
    upper_total = jnp.sum(filled[..., :6], axis=-1)
    bonus = jnp.where(upper_total >= c.UPPER_BONUS_THRESHOLD, c.UPPER_BONUS, 0)
    return jnp.sum(filled, axis=-1) + bonus
