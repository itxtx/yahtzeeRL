# YahtzeeRL

JAX/Flax/RLax Yahtzee self-play experiments with MCTS.

The current implementation targets two-player head-to-head self-play using
simplified standard Yahtzee scoring: 13 categories plus upper bonus, without
Joker rules or extra Yahtzee bonuses.

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
sign(score_margin) + 0.25 * tanh(score_margin / 50)
```

Use pure win/loss targets for comparison:

```bash
python -m yahtzee_rl.train --reward-mode win_loss
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
