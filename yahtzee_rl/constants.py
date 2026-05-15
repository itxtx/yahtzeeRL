"""Shared Yahtzee constants."""

NUM_DICE = 5
NUM_FACES = 6
NUM_PLAYERS = 2
NUM_CATEGORIES = 13
NUM_HOLD_ACTIONS = 2**NUM_DICE
NUM_ACTIONS = NUM_HOLD_ACTIONS + NUM_CATEGORIES
MAX_ROLLS_LEFT = 2
UPPER_BONUS_THRESHOLD = 63
UPPER_BONUS = 35

ONES = 0
TWOS = 1
THREES = 2
FOURS = 3
FIVES = 4
SIXES = 5
THREE_OF_A_KIND = 6
FOUR_OF_A_KIND = 7
FULL_HOUSE = 8
SMALL_STRAIGHT = 9
LARGE_STRAIGHT = 10
YAHTZEE = 11
CHANCE = 12

CATEGORY_NAMES = (
    "ones",
    "twos",
    "threes",
    "fours",
    "fives",
    "sixes",
    "three_of_a_kind",
    "four_of_a_kind",
    "full_house",
    "small_straight",
    "large_straight",
    "yahtzee",
    "chance",
)
