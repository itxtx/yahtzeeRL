import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.scoring import score_categories, total_score


def test_scores_full_house_and_yahtzee_options():
    scores = score_categories(jnp.array([[5, 5, 5, 2, 2]]))[0]
    assert int(scores[c.FIVES]) == 15
    assert int(scores[c.THREE_OF_A_KIND]) == 19
    assert int(scores[c.FULL_HOUSE]) == 25
    assert int(scores[c.YAHTZEE]) == 0
    assert int(scores[c.CHANCE]) == 19


def test_scores_straights():
    scores = score_categories(jnp.array([[1, 2, 3, 4, 6], [2, 3, 4, 5, 6]]))
    assert int(scores[0, c.SMALL_STRAIGHT]) == 30
    assert int(scores[0, c.LARGE_STRAIGHT]) == 0
    assert int(scores[1, c.SMALL_STRAIGHT]) == 30
    assert int(scores[1, c.LARGE_STRAIGHT]) == 40


def test_total_score_adds_upper_bonus():
    scorecard = jnp.array([[3, 6, 9, 12, 15, 18, -1, -1, -1, -1, -1, -1, -1]])
    assert int(total_score(scorecard)[0]) == 98
