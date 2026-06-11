"""Precomputed tables for dice multisets and exact chance distributions.

A chance outcome is a complete sorted 5-dice hand. There are
C(6 + 5 - 1, 5) = 252 distinct hands. Given the multiset of kept dice, the
probability of each final hand is an exact multinomial over the rerolled dice,
so the search can marginalize over chance correctly instead of sampling.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from yahtzee_rl import constants as c

_HANDS = sorted(itertools.combinations_with_replacement(range(1, c.NUM_FACES + 1), c.NUM_DICE))
NUM_CHANCE_OUTCOMES = len(_HANDS)  # 252

# (252, 5) sorted dice values for each chance outcome.
DICE_TABLE = jnp.asarray(np.array(_HANDS, dtype=np.int32))

# (252, 6) per-face counts for each chance outcome.
COUNTS_TABLE = jnp.asarray(
    np.array(
        [[hand.count(face) for face in range(1, c.NUM_FACES + 1)] for hand in _HANDS],
        dtype=np.int32,
    )
)

_LOG_FACTORIAL = jnp.asarray(
    np.log([math.factorial(i) for i in range(c.NUM_DICE + 1)]).astype(np.float32)
)
_LOG_NUM_FACES = float(np.log(c.NUM_FACES))


def chance_log_probs(kept_counts: jax.Array) -> jax.Array:
    """Exact log P(final sorted hand | kept dice) shaped (batch, 252).

    Args:
        kept_counts: (batch, 6) integer per-face counts of kept dice, sum <= 5.
            The remaining ``5 - sum`` dice are rerolled uniformly.

    Returns:
        (batch, 252) log-probabilities; unreachable hands get -inf. Each row is
        a normalized distribution (multinomial over the rerolled dice).
    """
    kept = kept_counts.astype(jnp.int32)
    num_rerolled = c.NUM_DICE - jnp.sum(kept, axis=-1)  # (batch,)
    rerolled = COUNTS_TABLE[None, :, :] - kept[:, None, :]  # (batch, 252, 6)
    reachable = jnp.all(rerolled >= 0, axis=-1)  # (batch, 252)
    rerolled_safe = jnp.clip(rerolled, 0, c.NUM_DICE)
    log_probs = (
        _LOG_FACTORIAL[num_rerolled][:, None]
        - jnp.sum(_LOG_FACTORIAL[rerolled_safe], axis=-1)
        - num_rerolled[:, None].astype(jnp.float32) * _LOG_NUM_FACES
    )
    return jnp.where(reachable, log_probs, -jnp.inf)
