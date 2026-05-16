"""Readable action helpers and simple heuristic policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.env import EnvState, current_scorecard, legal_action_mask
from yahtzee_rl.scoring import dice_counts, score_categories


def hold_mask_for_action(action: int) -> list[bool]:
    return [bool((action >> idx) & 1) for idx in range(c.NUM_DICE)]


def hold_action_for_face(dice: jax.Array, face: jax.Array) -> jax.Array:
    """Return hold action ids that keep every die equal to face."""
    bits = (dice == face[:, None]).astype(jnp.int32)
    powers = 2 ** jnp.arange(c.NUM_DICE, dtype=jnp.int32)
    return jnp.sum(bits * powers[None, :], axis=-1)


def action_label(action: int, dice: list[int] | None = None, scores: list[int] | None = None) -> str:
    """Return a compact human-readable action label."""
    if action < c.NUM_HOLD_ACTIONS:
        if dice is None:
            return f"h{action:02d} hold_mask={action:05b}"
        kept = [str(die) for die, keep in zip(dice, hold_mask_for_action(action)) if keep]
        detail = "keep " + " ".join(kept) if kept else "reroll all dice"
        return f"h{action:02d} {detail}"

    category = action - c.NUM_HOLD_ACTIONS
    score_text = ""
    if scores is not None:
        score_text = f" ({scores[category]} pts)"
    return f"s{category:02d} score {c.CATEGORY_NAMES[category]}{score_text}"


def immediate_score_for_action(action: int, scores: list[int]) -> int | None:
    if action < c.NUM_HOLD_ACTIONS:
        return None
    return scores[action - c.NUM_HOLD_ACTIONS]


def heuristic_action(state: EnvState) -> jax.Array:
    """Batched deterministic heuristic baseline.

    The heuristic takes clearly strong completed categories, scores the best
    legal category when forced, otherwise keeps the most frequent die face with
    higher faces winning ties.
    """
    legal = legal_action_mask(state)
    scores = score_categories(state.dice)
    score_legal = legal[:, c.NUM_HOLD_ACTIONS :]
    legal_scores = jnp.where(score_legal, scores, -1)

    best_score_category = jnp.argmax(legal_scores, axis=-1)
    best_score_action = c.NUM_HOLD_ACTIONS + best_score_category

    counts = dice_counts(state.dice)
    weighted_counts = counts * 10 + jnp.arange(1, c.NUM_FACES + 1)[None, :]
    face = jnp.argmax(weighted_counts, axis=-1).astype(jnp.int32) + 1
    hold_action = hold_action_for_face(state.dice, face)

    current = current_scorecard(state)
    yahtzee_action = jnp.full_like(best_score_action, c.NUM_HOLD_ACTIONS + c.YAHTZEE)
    large_straight_action = jnp.full_like(
        best_score_action, c.NUM_HOLD_ACTIONS + c.LARGE_STRAIGHT
    )
    full_house_action = jnp.full_like(best_score_action, c.NUM_HOLD_ACTIONS + c.FULL_HOUSE)

    can_yahtzee = (current[:, c.YAHTZEE] < 0) & (scores[:, c.YAHTZEE] == 50)
    can_large = (current[:, c.LARGE_STRAIGHT] < 0) & (scores[:, c.LARGE_STRAIGHT] == 40)
    can_full_house = (current[:, c.FULL_HOUSE] < 0) & (scores[:, c.FULL_HOUSE] == 25)
    forced_score = state.rolls_left == 0

    action = jnp.where(forced_score, best_score_action, hold_action)
    action = jnp.where(can_full_house & (~forced_score), full_house_action, action)
    action = jnp.where(can_large & (~forced_score), large_straight_action, action)
    action = jnp.where(can_yahtzee & (~forced_score), yahtzee_action, action)
    return action.astype(jnp.int32)
