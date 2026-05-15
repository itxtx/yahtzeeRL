"""Small random-policy evaluation utility for the environment."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from yahtzee_rl.env import legal_action_mask, reset, step
from yahtzee_rl.scoring import total_score
from yahtzee_rl.self_play import MAX_GAME_STEPS


def random_games(seed: int, batch_size: int):
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, batch_size)

    for _ in range(MAX_GAME_STEPS):
        key, action_key, step_key = jax.random.split(key, 3)
        mask = legal_action_mask(state)
        logits = jnp.where(mask, 0.0, jnp.finfo(jnp.float32).min)
        action = jax.random.categorical(action_key, logits).astype(jnp.int32)
        state, _ = step(state, action, step_key)

    return total_score(state.scorecards)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    scores = random_games(args.seed, args.batch_size)
    print(f"mean player 0 score: {float(jnp.mean(scores[:, 0])):.2f}")
    print(f"mean player 1 score: {float(jnp.mean(scores[:, 1])):.2f}")
    print(f"draw rate: {float(jnp.mean(scores[:, 0] == scores[:, 1])):.3f}")


if __name__ == "__main__":
    main()
