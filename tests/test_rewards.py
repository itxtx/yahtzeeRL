import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.env import EnvState
from yahtzee_rl.rewards import terminal_values_from_scores
from yahtzee_rl.self_play import terminal_values


def make_scorecards(player0_chance: int, player1_chance: int):
    scorecards = jnp.zeros((1, c.NUM_PLAYERS, c.NUM_CATEGORIES), dtype=jnp.int32)
    scorecards = scorecards.at[0, 0, c.CHANCE].set(player0_chance)
    scorecards = scorecards.at[0, 1, c.CHANCE].set(player1_chance)
    return scorecards


def test_win_loss_reward_and_perspective_flip():
    scorecards = make_scorecards(20, 10)

    values = terminal_values_from_scores(
        scorecards,
        jnp.array([0]),
        reward_mode="win_loss",
    )
    flipped = terminal_values_from_scores(
        scorecards,
        jnp.array([1]),
        reward_mode="win_loss",
    )

    assert float(values[0]) == 1.0
    assert float(flipped[0]) == -1.0


def test_draw_reward_is_zero():
    scorecards = make_scorecards(10, 10)
    values = terminal_values_from_scores(
        scorecards,
        jnp.array([0]),
        reward_mode="win_loss_margin",
    )

    assert float(values[0]) == 0.0


def test_margin_reward_increases_with_margin():
    small = make_scorecards(20, 10)
    large = make_scorecards(60, 10)

    small_value = terminal_values_from_scores(
        small,
        jnp.array([0]),
        reward_mode="win_loss_margin",
        margin_weight=0.25,
        margin_scale=50.0,
    )
    large_value = terminal_values_from_scores(
        large,
        jnp.array([0]),
        reward_mode="win_loss_margin",
        margin_weight=0.25,
        margin_scale=50.0,
    )

    # Shaped reward stays within [-1, 1] (matching the tanh value head) while
    # remaining monotonic in the score margin.
    assert 1.0 >= float(large_value[0]) > float(small_value[0]) > 0.0


def test_self_play_terminal_values_shape_and_perspective():
    scorecards = make_scorecards(20, 10)
    state = EnvState(
        scorecards=scorecards,
        dice=jnp.array([[1, 1, 1, 1, 1]]),
        rolls_left=jnp.array([0]),
        active_player=jnp.array([0]),
        done=jnp.array([True]),
    )
    acting_players = jnp.array([[0], [1]])

    values = terminal_values(state, acting_players, reward_mode="win_loss")

    assert values.shape == acting_players.shape
    assert float(values[0, 0]) == 1.0
    assert float(values[1, 0]) == -1.0
