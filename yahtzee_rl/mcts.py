"""Stochastic MuZero search over the real Yahtzee transition.

The search uses explicit decision and chance nodes (mctx.stochastic_muzero_policy)
instead of sampling one dice outcome per tree edge:

  decision node (real EnvState) --action--> afterstate --chance outcome--> state

An afterstate is the deterministic part of an action: the kept dice after a
hold, or the updated scorecard after scoring. A chance outcome is the complete
sorted 5-dice hand that results from the pending reroll; its exact multinomial
distribution is computed analytically (dice_tables.chance_log_probs), so the
search marginalizes over chance correctly.

Two-player negamax convention: mctx hardcodes reward 0 / discount +1 on
decision->afterstate edges, so afterstate values are from the acting player's
perspective. Perspective flips happen on chance->state edges: discount is -1
when scoring passed the turn, +1 when the same player continues after a hold,
and 0 into terminal states (with the terminal reward delivered on that edge,
from the acting player's perspective).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx

from yahtzee_rl import constants as c
from yahtzee_rl.dice_tables import DICE_TABLE, chance_log_probs
from yahtzee_rl.env import (
    EnvState,
    action_to_hold_mask,
    encode_observation,
    legal_action_mask,
    observation,
)
from yahtzee_rl.model import masked_logits
from yahtzee_rl.rewards import WIN_LOSS_MARGIN, terminal_value_from_state
from yahtzee_rl.scoring import dice_counts, score_category


class Afterstate(NamedTuple):
    """Deterministic result of an action, before the pending dice reroll.

    acting_player is the player who took the action (the perspective of the
    afterstate value); next_player is who moves once chance resolves.
    """

    scorecards: jax.Array  # (batch, 2, 13)
    kept_counts: jax.Array  # (batch, 6) per-face counts of dice not rerolled
    rolls_left: jax.Array  # (batch,) rolls left once chance resolves
    acting_player: jax.Array  # (batch,)
    next_player: jax.Array  # (batch,)
    done: jax.Array  # (batch,)


def _predict(model, params, state: EnvState) -> tuple[jax.Array, jax.Array]:
    obs = observation(state)
    logits, value = model.apply(params, obs)
    return masked_logits(logits, legal_action_mask(state)), value


def afterstate_observation(after: Afterstate) -> dict[str, jax.Array]:
    """Encode an afterstate from the acting player's perspective."""
    num_unknown = c.NUM_DICE - jnp.sum(after.kept_counts, axis=-1)
    return encode_observation(
        scorecards=after.scorecards,
        perspective_player=after.acting_player,
        dice_count_vec=after.kept_counts,
        num_unknown=num_unknown,
        rolls_left=after.rolls_left,
        opponent_to_move=(after.next_player != after.acting_player),
    )


def _replace_active_score(
    scorecards: jax.Array, active_player: jax.Array, category: jax.Array, score: jax.Array
) -> jax.Array:
    batch = jnp.arange(scorecards.shape[0])
    return scorecards.at[batch, active_player, category].set(score)


def make_decision_recurrent_fn(
    model,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
):
    """Decision node expansion: apply the deterministic part of an action."""

    def decision_fn(params, rng_key, action, state: EnvState):
        del rng_key  # The decision step is deterministic.
        # mctx calls this fn for every expansion (including chance slots), so
        # the incoming index may be out of the action range: clip defensively.
        action = jnp.clip(action.astype(jnp.int32), 0, c.NUM_ACTIONS - 1)
        is_score = action >= c.NUM_HOLD_ACTIONS

        hold_mask = action_to_hold_mask(jnp.clip(action, 0, c.NUM_HOLD_ACTIONS - 1))
        # Dice value 0 falls outside one_hot range and contributes nothing.
        kept_after_hold = dice_counts(jnp.where(hold_mask, state.dice, 0))

        category = jnp.clip(action - c.NUM_HOLD_ACTIONS, 0, c.NUM_CATEGORIES - 1)
        scored_value = score_category(state.dice, category)
        scorecards_scored = _replace_active_score(
            state.scorecards, state.active_player, category, scored_value
        )
        effective_score = is_score & (~state.done)
        scorecards_after = jnp.where(
            effective_score[:, None, None], scorecards_scored, state.scorecards
        )
        done_after = state.done | (
            effective_score & jnp.all(scorecards_scored >= 0, axis=(1, 2))
        )

        # Hold keeps the held dice; scoring hands all five dice to the next
        # player; terminal afterstates keep the full current hand so the only
        # reachable chance outcome is the hand itself (probability one).
        full_counts = dice_counts(state.dice)
        kept_counts = jnp.where(effective_score[:, None], 0, kept_after_hold)
        kept_counts = jnp.where(done_after[:, None], full_counts, kept_counts)

        rolls_left = jnp.where(
            effective_score,
            c.MAX_ROLLS_LEFT,
            jnp.maximum(state.rolls_left - 1, 0),
        )
        rolls_left = jnp.where(done_after, state.rolls_left, rolls_left)
        next_player = jnp.where(
            effective_score & (~done_after),
            1 - state.active_player,
            state.active_player,
        )

        afterstate = Afterstate(
            scorecards=scorecards_after,
            kept_counts=kept_counts.astype(jnp.int32),
            rolls_left=rolls_left.astype(jnp.int32),
            acting_player=state.active_player,
            next_player=next_player.astype(jnp.int32),
            done=done_after,
        )

        _, net_value = model.apply(params, afterstate_observation(afterstate))
        terminal_value = terminal_value_from_state(
            scorecards_after,
            state.active_player,
            reward_mode=reward_mode,
            margin_weight=margin_weight,
            margin_scale=margin_scale,
        )
        afterstate_value = jnp.where(done_after, terminal_value, net_value)

        output = mctx.DecisionRecurrentFnOutput(
            chance_logits=chance_log_probs(kept_counts),
            afterstate_value=afterstate_value.astype(jnp.float32),
        )
        return output, afterstate

    return decision_fn


def make_chance_recurrent_fn(
    model,
    reward_mode: str = WIN_LOSS_MARGIN,
    margin_weight: float = 0.25,
    margin_scale: float = 50.0,
):
    """Chance node expansion: resolve the pending reroll into a real state."""

    def chance_fn(params, rng_key, chance_outcome, after: Afterstate):
        del rng_key  # Chance outcomes are enumerated, not sampled.
        # mctx calls this fn for every expansion (including decision slots), so
        # the incoming index may be negative: clip defensively.
        outcome = jnp.clip(chance_outcome.astype(jnp.int32), 0, DICE_TABLE.shape[0] - 1)
        next_state = EnvState(
            scorecards=after.scorecards,
            dice=DICE_TABLE[outcome],
            rolls_left=after.rolls_left,
            active_player=after.next_player,
            done=after.done,
        )
        action_logits, value = _predict(model, params, next_state)

        player_changed = after.next_player != after.acting_player
        discount = jnp.where(player_changed, -1.0, 1.0)
        discount = jnp.where(after.done, 0.0, discount).astype(jnp.float32)
        terminal_value = terminal_value_from_state(
            after.scorecards,
            after.acting_player,
            reward_mode=reward_mode,
            margin_weight=margin_weight,
            margin_scale=margin_scale,
        )
        reward = jnp.where(after.done, terminal_value, 0.0).astype(jnp.float32)

        output = mctx.ChanceRecurrentFnOutput(
            action_logits=action_logits,
            value=value.astype(jnp.float32),
            reward=reward,
            discount=discount,
        )
        return output, next_state

    return chance_fn


def root_output(model, params, state: EnvState) -> mctx.RootFnOutput:
    logits, value = _predict(model, params, state)
    return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)


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
    eval_mode: bool = False,
) -> mctx.PolicyOutput:
    """Run batched Stochastic MuZero search and return the mctx policy output.

    eval_mode disables root Dirichlet noise and acts greedily with respect to
    visit counts; training mode keeps exploration noise and samples actions
    proportional to visit counts at the given temperature.
    """
    root = root_output(model, params, state)
    return mctx.stochastic_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        decision_recurrent_fn=make_decision_recurrent_fn(
            model,
            reward_mode=reward_mode,
            margin_weight=margin_weight,
            margin_scale=margin_scale,
        ),
        chance_recurrent_fn=make_chance_recurrent_fn(
            model,
            reward_mode=reward_mode,
            margin_weight=margin_weight,
            margin_scale=margin_scale,
        ),
        num_simulations=num_simulations,
        invalid_actions=(~legal_action_mask(state)).astype(jnp.float32),
        dirichlet_fraction=0.0 if eval_mode else 0.25,
        temperature=0.0 if eval_mode else temperature,
    )
