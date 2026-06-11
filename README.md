# YahtzeeRL

JAX/Flax/RLax Yahtzee self-play experiments with MCTS.

The current implementation targets two-player head-to-head self-play using
simplified standard Yahtzee scoring: 13 categories plus upper bonus, without
Joker rules or extra Yahtzee bonuses.

The search uses Stochastic MuZero (`mctx.stochastic_muzero_policy`) with
explicit decision and chance nodes: actions resolve to afterstates, and the
pending dice reroll is a chance node over the 252 sorted 5-dice hands with its
exact multinomial distribution computed analytically. Dice are kept sorted in
the environment (game-equivalent), so permutation-equivalent states collapse.
Training samples minibatches from a replay buffer of recent self-play frames
and regresses values onto a blend of the terminal outcome and the search root
value (`--value-target-outcome-weight`). Evaluation and CLI play run the search
without Dirichlet noise and act greedily on visit counts.

## Local CPU Smoke Tests

```bash
python -m pytest
python -m yahtzee_rl.evaluate --batch-size 16
```

## Training

Local Apple Silicon JAX runs on CPU, so use small settings locally:

```bash
python -m yahtzee_rl.train --steps 2 --batch-size 2 --num-simulations 2
```

Training defaults to margin-shaped terminal rewards:

```text
sign(score_margin) * (1 - 0.25) + 0.25 * tanh(score_margin / 50)
```

This stays within `[-1, 1]` to match the tanh-bounded value head, splitting the
unit reward between a win/loss baseline and a margin-sensitive term.

Use pure win/loss targets for comparison:

```bash
python -m yahtzee_rl.train --reward-mode win_loss
```

Replay and value-target knobs (defaults shown):

```bash
python -m yahtzee_rl.train \
  --buffer-size 100000 \
  --minibatches-per-update 4 \
  --minibatch-size 1024 \
  --value-target-outcome-weight 0.5
```

For GPU training, copy this repo to Google Drive and run:

```text
notebooks/colab_self_play.ipynb
```

The notebook mounts Drive, installs the repo, verifies `jax.devices()`, and
copies the repo to `/content/yahtzeeRL` for faster Colab execution while writing
checkpoints back to Drive.

The Colab notebook runs full training/evaluation through `python -m ...`
subprocesses instead of `train(config)` inside the notebook kernel. This avoids
keeping JAX's GPU allocator pool and compiled programs alive after training
finishes.

## Play Against The Agent

```bash
python -m yahtzee_rl.play_cli --checkpoint checkpoints/colab_run --num-simulations 32
python -m yahtzee_rl.play_cli --checkpoint checkpoints/colab_run --debug-agent --top-k 5
```

During your turn, choose reroll actions as `h00` through `h31` or score actions
as `s00` through `s12`. The CLI prints the legal actions each turn.

## Evaluate Agents

```bash
python -m yahtzee_rl.evaluate --agent-a mcts --checkpoint-a checkpoints/colab_run --agent-b heuristic --num-games 256 --sims-a 32
python -m yahtzee_rl.evaluate --agent-a mcts --checkpoint-a checkpoints/run_a --agent-b mcts --checkpoint-b checkpoints/run_b --num-games 256
python -m yahtzee_rl.evaluate --num-games 16
```

The evaluator alternates seats so first-player effects do not dominate the
reported win rate, score margin, and optional per-category means.
