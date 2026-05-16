import jax
import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.actions import action_label, heuristic_action
from yahtzee_rl.env import reset


def test_action_label_formats_hold_and_score():
    dice = [4, 5, 1, 4, 2]
    scores = [0] * c.NUM_CATEGORIES
    scores[c.FOUR_OF_A_KIND] = 0

    assert action_label(3, dice) == "h03 keep 4 5"
    assert (
        action_label(c.NUM_HOLD_ACTIONS + c.FOUR_OF_A_KIND, dice, scores)
        == "s07 score four_of_a_kind (0 pts)"
    )


def test_heuristic_forced_scoring_takes_highest_immediate_score():
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    state = state._replace(dice=jnp.array([[4, 4, 4, 5, 6]]), rolls_left=jnp.array([0]))

    action = heuristic_action(state)

    assert int(action[0]) == c.NUM_HOLD_ACTIONS + c.THREE_OF_A_KIND


def test_heuristic_with_rolls_left_keeps_most_frequent_high_face():
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    state = state._replace(dice=jnp.array([[4, 5, 1, 4, 2]]), rolls_left=jnp.array([2]))

    action = heuristic_action(state)

    assert int(action[0]) == 9


def test_heuristic_55522_scores_full_house_or_keeps_fives():
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    state = state._replace(dice=jnp.array([[5, 5, 5, 2, 2]]), rolls_left=jnp.array([1]))
    assert int(heuristic_action(state)[0]) == c.NUM_HOLD_ACTIONS + c.FULL_HOUSE

    scorecards = state.scorecards.at[0, 0, c.FULL_HOUSE].set(25)
    state = state._replace(scorecards=scorecards)
    assert int(heuristic_action(state)[0]) == 7
