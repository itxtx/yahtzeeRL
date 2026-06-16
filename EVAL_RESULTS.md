# Evaluation Results

Collected evaluation notes for the competitive head-to-head Yahtzee agent.
Unless noted otherwise, checkpoints refer to the
`win_loss_margin_32simsrun4` run.

## Summary

`step_011800` is the best competitive checkpoint observed so far. Continuing
training to `step_012800` regressed the greedy policy in direct comparison.

## Competitive Baseline

```text
Agent A: greedy@step_12800
Agent B: heuristic
games: 512
A win: 0.781 | B win: 0.217 | draw: 0.002
mean score A: 202.77 | B: 165.38 | margin: 37.38

Per-category means
Category                    A        B
ones                     1.36     1.10
twos                     4.08     3.50
threes                   7.05     5.79
fours                    9.12     8.13
fives                   12.81    10.25
sixes                   14.88    13.37
three_of_a_kind         19.76    18.68
four_of_a_kind          14.77    17.81
full_house              22.12    23.05
small_straight          29.41     7.97
large_straight          34.14    12.50
yahtzee                  9.86    23.63
chance                  21.96    18.98
```

## Search Versus Greedy Policy

The following evaluations compare MCTS at `step_011800` against the same
checkpoint's greedy policy head. The search advantage is essentially flat,
suggesting that the policy head has already absorbed the useful shallow/medium
search target.

```text
Agent A: mcts@step_11800
Agent B: greedy@step_11800
games: 256
A win: 0.473 | B win: 0.516 | draw: 0.012
mean score A: 204.19 | B: 207.86 | margin: -3.67
```

```text
Agent A: mcts@step_11800
Agent B: greedy@step_11800
games: 512
A win: 0.494 | B win: 0.492 | draw: 0.014
mean score A: 208.17 | B: 207.21 | margin: 0.96
```

```text
Agent A: mcts@step_11800
Agent B: greedy@step_11800
games: 512
A win: 0.525 | B win: 0.471 | draw: 0.004
mean score A: 210.00 | B: 209.74 | margin: 0.26
```

```text
Agent A: mcts@step_11800
Agent B: greedy@step_11800
games: 512
A win: 0.508 | B win: 0.486 | draw: 0.006
mean score A: 210.07 | B: 210.65 | margin: -0.58
```

The three 512-game runs above correspond to MCTS sims 64, 128, and 256 versus
greedy, respectively.

## Checkpoint Regression

```text
Agent A: greedy@step_11800
Agent B: greedy@step_12800
games: 512
A win: 0.553 | B win: 0.436 | draw: 0.012
mean score A: 209.03 | B: 201.90 | margin: 7.13
```

This direct comparison is the main reason to preserve `step_011800` as the best
competitive checkpoint and avoid treating `step_012800` as the new default.
