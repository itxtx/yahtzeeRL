import jax
import jax.numpy as jnp
import numpy as np

from yahtzee_rl import constants as c
from yahtzee_rl.dice_tables import COUNTS_TABLE, DICE_TABLE, NUM_CHANCE_OUTCOMES, chance_log_probs
from yahtzee_rl.env import EnvState, legal_action_mask, observation, reset, step
from yahtzee_rl.mcts import (
    make_chance_recurrent_fn,
    make_decision_recurrent_fn,
    search_policy,
)
from yahtzee_rl.model import YahtzeeActorCritic


def make_model_and_params(batch_size=1):
    model = YahtzeeActorCritic(hidden_dims=(8, 8))
    state = reset(jax.random.PRNGKey(0), batch_size)
    params = model.init(jax.random.PRNGKey(1), observation(state))
    return model, params


def test_dice_tables_shapes():
    assert NUM_CHANCE_OUTCOMES == 252
    assert DICE_TABLE.shape == (252, c.NUM_DICE)
    assert COUNTS_TABLE.shape == (252, c.NUM_FACES)
    assert bool(jnp.all(jnp.sum(COUNTS_TABLE, axis=-1) == c.NUM_DICE))
    # Hands are sorted.
    assert bool(jnp.all(DICE_TABLE[:, :-1] <= DICE_TABLE[:, 1:]))


def test_chance_log_probs_normalized_and_supported():
    kept = jnp.array(
        [
            [0, 0, 0, 0, 0, 0],  # reroll all five
            [0, 0, 3, 0, 0, 0],  # keep three 3s
            [1, 1, 1, 1, 1, 0],  # keep small straight 1-5
            [0, 0, 0, 0, 0, 5],  # keep everything
        ]
    )
    log_probs = chance_log_probs(kept)
    probs = jnp.where(jnp.isfinite(log_probs), jnp.exp(log_probs), 0.0)
    np.testing.assert_allclose(np.asarray(jnp.sum(probs, axis=-1)), 1.0, rtol=1e-5)

    # Support: only hands containing the kept dice are reachable.
    reachable = jnp.isfinite(log_probs)
    contains_kept = jnp.all(COUNTS_TABLE[None, :, :] >= kept[:, None, :], axis=-1)
    assert bool(jnp.all(reachable == contains_kept))

    # Keeping all five dice leaves exactly one outcome with probability one.
    assert int(jnp.sum(reachable[3])) == 1
    assert float(jnp.max(probs[3])) == 1.0


def _terminal_ready_state(dice, active=0):
    """All categories filled with 0 except the active player's CHANCE."""
    scorecards = jnp.zeros((1, c.NUM_PLAYERS, c.NUM_CATEGORIES), dtype=jnp.int32)
    scorecards = scorecards.at[0, active, c.CHANCE].set(-1)
    return EnvState(
        scorecards=scorecards,
        dice=jnp.array([dice]),
        rolls_left=jnp.array([0]),
        active_player=jnp.array([active]),
        done=jnp.array([False]),
    )


def test_decision_fn_hold_and_score_transitions():
    model, params = make_model_and_params()
    decision_fn = make_decision_recurrent_fn(model, reward_mode="win_loss")
    state = reset(jax.random.PRNGKey(0), 1)
    state = state._replace(dice=jnp.array([[2, 3, 5, 5, 6]]), rolls_left=jnp.array([2]))

    # Hold action keeping both fives: positions 2 and 3 -> bits 4 + 8 = 12.
    out, after = decision_fn(params, jax.random.PRNGKey(0), jnp.array([12]), state)
    assert after.kept_counts[0].tolist() == [0, 0, 0, 0, 2, 0]
    assert int(after.rolls_left[0]) == 1
    assert int(after.next_player[0]) == int(after.acting_player[0])
    assert not bool(after.done[0])
    probs = jnp.exp(out.chance_logits[0])
    np.testing.assert_allclose(float(jnp.sum(jnp.where(jnp.isfinite(out.chance_logits[0]), probs, 0.0))), 1.0, rtol=1e-5)

    # Non-terminal scoring passes the turn and rerolls all five dice.
    out, after = decision_fn(
        params, jax.random.PRNGKey(0), jnp.array([c.NUM_HOLD_ACTIONS + c.CHANCE]), state
    )
    assert int(jnp.sum(after.kept_counts[0])) == 0
    assert int(after.rolls_left[0]) == c.MAX_ROLLS_LEFT
    assert int(after.next_player[0]) == 1
    assert not bool(after.done[0])


def test_decision_fn_terminal_afterstate_value_is_exact():
    model, params = make_model_and_params()
    decision_fn = make_decision_recurrent_fn(model, reward_mode="win_loss")
    state = _terminal_ready_state([6, 6, 6, 6, 6])

    out, after = decision_fn(
        params, jax.random.PRNGKey(0), jnp.array([c.NUM_HOLD_ACTIONS + c.CHANCE]), state
    )
    assert bool(after.done[0])
    # Player 0 totals 30 vs 0: exact terminal value, not a network estimate.
    assert float(out.afterstate_value[0]) == 1.0
    # Terminal afterstates concentrate chance on the current hand.
    probs = jnp.exp(out.chance_logits[0])
    reachable = jnp.isfinite(out.chance_logits[0])
    assert int(jnp.sum(reachable)) == 1
    assert float(jnp.max(jnp.where(reachable, probs, 0.0))) == 1.0


def test_chance_fn_negamax_signs():
    """Discount must flip only when the turn passes, and zero at terminal."""
    model, params = make_model_and_params()
    decision_fn = make_decision_recurrent_fn(model, reward_mode="win_loss")
    chance_fn = make_chance_recurrent_fn(model, reward_mode="win_loss")
    outcome = jnp.array([0])

    # Hold: same player continues -> discount +1, reward 0.
    state = reset(jax.random.PRNGKey(0), 1)
    _, after = decision_fn(params, jax.random.PRNGKey(0), jnp.array([0]), state)
    out, next_state = chance_fn(params, jax.random.PRNGKey(0), outcome, after)
    assert float(out.discount[0]) == 1.0
    assert float(out.reward[0]) == 0.0
    assert int(next_state.active_player[0]) == int(state.active_player[0])

    # Non-terminal score: turn passes -> discount -1, reward 0.
    _, after = decision_fn(
        params, jax.random.PRNGKey(0), jnp.array([c.NUM_HOLD_ACTIONS + c.ONES]), state
    )
    out, next_state = chance_fn(params, jax.random.PRNGKey(0), outcome, after)
    assert float(out.discount[0]) == -1.0
    assert float(out.reward[0]) == 0.0
    assert int(next_state.active_player[0]) == 1 - int(state.active_player[0])

    # Terminal score: discount 0 and the exact terminal reward from the
    # acting player's perspective is delivered on the chance edge.
    state = _terminal_ready_state([6, 6, 6, 6, 6])
    _, after = decision_fn(
        params, jax.random.PRNGKey(0), jnp.array([c.NUM_HOLD_ACTIONS + c.CHANCE]), state
    )
    out, next_state = chance_fn(params, jax.random.PRNGKey(0), outcome, after)
    assert float(out.discount[0]) == 0.0
    assert float(out.reward[0]) == 1.0
    assert bool(next_state.done[0])

    # Same terminal position but the loser acts: reward flips sign.
    state = _terminal_ready_state([6, 6, 6, 6, 6], active=1)
    scorecards = state.scorecards.at[0, 0, c.CHANCE].set(50)
    state = state._replace(scorecards=scorecards, dice=jnp.array([[1, 1, 1, 1, 1]]))
    _, after = decision_fn(
        params, jax.random.PRNGKey(0), jnp.array([c.NUM_HOLD_ACTIONS + c.CHANCE]), state
    )
    out, _ = chance_fn(params, jax.random.PRNGKey(0), outcome, after)
    assert float(out.discount[0]) == 0.0
    assert float(out.reward[0]) == -1.0


def test_search_policy_returns_legal_actions_and_decision_weights():
    model, params = make_model_and_params(batch_size=3)
    state = reset(jax.random.PRNGKey(3), 3)
    out = search_policy(model, params, state, jax.random.PRNGKey(4), num_simulations=8)

    assert out.action_weights.shape == (3, c.NUM_ACTIONS)
    np.testing.assert_allclose(np.asarray(jnp.sum(out.action_weights, axis=-1)), 1.0, rtol=1e-5)
    legal = legal_action_mask(state)
    for i in range(3):
        assert bool(legal[i, out.action[i]])
    # Root summary value is available for blended value targets.
    assert out.search_tree.summary().value.shape == (3,)


def test_env_dice_stay_sorted():
    key = jax.random.PRNGKey(0)
    state = reset(key, 8)
    assert bool(jnp.all(state.dice[:, :-1] <= state.dice[:, 1:]))
    for i in range(12):
        key, akey, skey = jax.random.split(key, 3)
        mask = legal_action_mask(state)
        logits = jnp.where(mask, 0.0, -1e9)
        action = jax.random.categorical(akey, logits)
        state, _ = step(state, action, skey)
        assert bool(jnp.all(state.dice[:, :-1] <= state.dice[:, 1:]))
