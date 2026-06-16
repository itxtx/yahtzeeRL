# Evaluation Results

Collected evaluation notes for the competitive head-to-head Yahtzee agent.
Unless noted otherwise, checkpoints refer to the
`win_loss_margin_32simsrun4` run.

## Summary

`step_011800` is the best competitive checkpoint observed so far. Continuing
training to `step_012800` regressed the greedy policy in direct comparison.

## Competitive Baseline

### Reproducibility Receipt

Best checkpoint:

```text
win_loss_margin_32simsrun4/step_011800
```

Recorded training segment leading into the headline checkpoint, reconstructed
from the retained training log:

```bash
python -m yahtzee_rl.train \
  --resume {CHECKPOINT_ROOT}/win_loss_margin_32simsrun4/step_010900 \
  --checkpoint-dir {CHECKPOINT_ROOT}/win_loss_margin_32simsrun4 \
  --steps 1000 \
  --batch-size 32 \
  --num-simulations 64 \
  --minibatches-per-update 1 \
  --teacher-every 3 \
  --teacher-batch-size 12 \
  --teacher-num-simulations 384 \
  --teacher-minibatches-per-update 3 \
  --buffer-size 50000 \
  --seed 0
```

Runtime receipt:

```text
Hardware/runtime: Google Colab CUDA runtime; JAX reported [CudaDevice(id=0)]
Seed: 0
Training duration: exact final duration was not retained. The partial log shows
  274/1000 updates after 2:28:15 for this segment; step_011800 was the 900th
  update after resuming from step_010900.
```

Headline evaluation command:

```bash
python -m yahtzee_rl.evaluate \
  --agent-a mcts \
  --checkpoint-a {CHECKPOINT_ROOT}/win_loss_margin_32simsrun4/step_011800 \
  --agent-b heuristic \
  --num-games 512 \
  --sims-a 32 \
  --per-category \
  --seed 0
```

Headline result:

```text
Agent A: mcts@step_11800
Agent B: heuristic
games: 512
A win: 0.811 | B win: 0.186 | draw: 0.004
mean score A: 204.28 | B: 164.48 | margin: 39.80

Per-category means
Category                    A        B
ones                     1.25     1.12
twos                     3.77     3.38
threes                   6.86     5.61
fours                    9.76     8.35
fives                   13.04    10.20
sixes                   15.89    13.12
three_of_a_kind         20.16    18.75
four_of_a_kind          15.98    17.31
full_house              22.02    23.10
small_straight          27.95     8.32
large_straight          33.20    12.66
yahtzee                  9.86    23.24
chance                  22.07    18.84
```

### Later Checkpoint Versus Heuristic

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
