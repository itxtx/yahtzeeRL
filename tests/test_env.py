import jax
import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.env import EnvState, legal_action_mask, reset, step


def test_roll_with_rolls_left_can_score_or_reroll():
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    state = state._replace(dice=jnp.array([[5, 5, 5, 2, 2]]), rolls_left=jnp.array([1]))
    mask = legal_action_mask(state)[0]

    assert bool(mask[0])
    assert bool(mask[c.NUM_HOLD_ACTIONS + c.FULL_HOUSE])
    assert bool(mask[c.NUM_HOLD_ACTIONS + c.FIVES])


def test_no_rolls_left_forces_scoring():
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    state = state._replace(rolls_left=jnp.array([0]))
    mask = legal_action_mask(state)[0]

    assert not bool(jnp.any(mask[: c.NUM_HOLD_ACTIONS]))
    assert bool(jnp.all(mask[c.NUM_HOLD_ACTIONS :]))


def test_scoring_switches_player_and_fills_category():
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    state = state._replace(dice=jnp.array([[5, 5, 5, 2, 2]]), rolls_left=jnp.array([1]))
    action = jnp.array([c.NUM_HOLD_ACTIONS + c.FULL_HOUSE])

    next_state, reward = step(state, action, jax.random.PRNGKey(1))

    assert int(next_state.scorecards[0, 0, c.FULL_HOUSE]) == 25
    assert int(next_state.active_player[0]) == 1
    assert int(next_state.rolls_left[0]) == c.MAX_ROLLS_LEFT
    assert float(reward[0]) == 0.0


def test_out_of_range_action_is_invalid_noop():
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    state = state._replace(dice=jnp.array([[5, 5, 5, 2, 2]]), rolls_left=jnp.array([2]))

    for action in (-1, c.NUM_ACTIONS, 999):
        next_state, reward = step(state, jnp.array([action]), jax.random.PRNGKey(1))
        assert float(reward[0]) == -1.0
        assert bool(jnp.all(next_state.scorecards == state.scorecards))
        assert int(next_state.rolls_left[0]) == int(state.rolls_left[0])
        assert int(next_state.active_player[0]) == int(state.active_player[0])


def test_terminal_winner_reward_is_from_actor_perspective():
    scorecards = jnp.zeros((1, c.NUM_PLAYERS, c.NUM_CATEGORIES), dtype=jnp.int32)
    scorecards = scorecards.at[0, 0, c.CHANCE].set(-1)
    state = EnvState(
        scorecards=scorecards,
        dice=jnp.array([[6, 6, 6, 6, 6]]),
        rolls_left=jnp.array([0]),
        active_player=jnp.array([0]),
        done=jnp.array([False]),
    )
    next_state, reward = step(
        state, jnp.array([c.NUM_HOLD_ACTIONS + c.CHANCE]), jax.random.PRNGKey(2)
    )

    assert bool(next_state.done[0])
    assert float(reward[0]) == 1.0
