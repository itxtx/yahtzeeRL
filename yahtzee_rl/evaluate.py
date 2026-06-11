"""Evaluate Yahtzee agents against random, heuristic, or checkpoint policies."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.actions import heuristic_action
from yahtzee_rl.env import EnvState, legal_action_mask, observation, reset, step
from yahtzee_rl.mcts import search_policy
from yahtzee_rl.model import masked_logits
from yahtzee_rl.rewards import WIN_LOSS_MARGIN
from yahtzee_rl.scoring import total_score
from yahtzee_rl.self_play import MAX_GAME_STEPS
from yahtzee_rl.train import load_checkpoint, load_checkpoint_config


@dataclass(frozen=True)
class PolicySpec:
    kind: str
    checkpoint: str | None = None
    num_simulations: int = 32
    params: object | None = None
    model: object | None = None
    step: int | None = None
    reward_mode: str = WIN_LOSS_MARGIN
    margin_weight: float = 0.25
    margin_scale: float = 50.0

    @property
    def label(self) -> str:
        if self.step is None:
            return self.kind
        return f"{self.kind}@step_{self.step}"


def load_policy(kind: str, checkpoint: str | None, num_simulations: int) -> PolicySpec:
    if kind in {"greedy", "mcts"}:
        if checkpoint is None:
            raise ValueError(f"--checkpoint is required for {kind} policy")
        config = load_checkpoint_config(checkpoint)
        state, model, step_idx = load_checkpoint(checkpoint)
        return PolicySpec(
            kind=kind,
            checkpoint=checkpoint,
            num_simulations=num_simulations,
            params=state.params,
            model=model,
            step=step_idx,
            reward_mode=config.reward_mode,
            margin_weight=config.margin_weight,
            margin_scale=config.margin_scale,
        )
    return PolicySpec(kind=kind, checkpoint=checkpoint, num_simulations=num_simulations)


def select_action(policy: PolicySpec, state: EnvState, key: jax.Array) -> jax.Array:
    if policy.kind == "random":
        mask = legal_action_mask(state)
        logits = jnp.where(mask, 0.0, jnp.finfo(jnp.float32).min)
        return jax.random.categorical(key, logits).astype(jnp.int32)

    if policy.kind == "heuristic":
        return heuristic_action(state)

    params = {"params": policy.params}
    if policy.kind == "greedy":
        logits, _ = policy.model.apply(params, observation(state))
        logits = masked_logits(logits, legal_action_mask(state))
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    if policy.kind == "mcts":
        output = search_policy(
            policy.model,
            params,
            state,
            key,
            num_simulations=policy.num_simulations,
            reward_mode=policy.reward_mode,
            margin_weight=policy.margin_weight,
            margin_scale=policy.margin_scale,
            eval_mode=True,
        )
        return output.action.astype(jnp.int32)

    raise ValueError(f"Unknown policy kind: {policy.kind}")


def run_games(
    policy_a: PolicySpec,
    policy_b: PolicySpec,
    seat_a: int,
    num_games: int,
    seed: int,
) -> dict[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, num_games)

    # Jit once per policy so the search is compiled instead of retraced on
    # every one of the MAX_GAME_STEPS python-loop iterations.
    action_a_fn = jax.jit(lambda s, k: select_action(policy_a, s, k))
    action_b_fn = jax.jit(lambda s, k: select_action(policy_b, s, k))
    step_fn = jax.jit(step)

    for _ in range(MAX_GAME_STEPS):
        key, key_a, key_b, step_key = jax.random.split(key, 4)
        action_a = action_a_fn(state, key_a)
        action_b = action_b_fn(state, key_b)
        use_a = state.active_player == seat_a
        action = jnp.where(use_a, action_a, action_b)
        state, _ = step_fn(state, action, step_key)

    scores = total_score(state.scorecards)
    a_score = scores[:, seat_a]
    b_score = scores[:, 1 - seat_a]
    margin = a_score - b_score
    return {
        "a_score": a_score,
        "b_score": b_score,
        "margin": margin,
        "a_win": margin > 0,
        "b_win": margin < 0,
        "draw": margin == 0,
        "a_scorecard": state.scorecards[:, seat_a, :],
        "b_scorecard": state.scorecards[:, 1 - seat_a, :],
    }


def concat_results(results: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    return {
        key: jnp.concatenate([result[key] for result in results], axis=0)
        for key in results[0]
    }


def evaluate_matchup(
    policy_a: PolicySpec,
    policy_b: PolicySpec,
    num_games: int,
    seed: int,
) -> dict[str, jax.Array]:
    games_a_first = (num_games + 1) // 2
    games_b_first = num_games // 2
    results = []
    if games_a_first:
        results.append(run_games(policy_a, policy_b, seat_a=0, num_games=games_a_first, seed=seed))
    if games_b_first:
        results.append(run_games(policy_a, policy_b, seat_a=1, num_games=games_b_first, seed=seed + 1))
    return concat_results(results)


def summarize(results: dict[str, jax.Array], per_category: bool = False) -> dict[str, float]:
    summary = {
        "games": int(results["a_score"].shape[0]),
        "a_win_rate": float(jnp.mean(results["a_win"])),
        "b_win_rate": float(jnp.mean(results["b_win"])),
        "draw_rate": float(jnp.mean(results["draw"])),
        "mean_a_score": float(jnp.mean(results["a_score"])),
        "mean_b_score": float(jnp.mean(results["b_score"])),
        "mean_margin": float(jnp.mean(results["margin"])),
    }
    if per_category:
        a_categories = jnp.mean(jnp.maximum(results["a_scorecard"], 0), axis=0)
        b_categories = jnp.mean(jnp.maximum(results["b_scorecard"], 0), axis=0)
        for idx, name in enumerate(c.CATEGORY_NAMES):
            summary[f"a_{name}"] = float(a_categories[idx])
            summary[f"b_{name}"] = float(b_categories[idx])
    return summary


def print_summary(policy_a: PolicySpec, policy_b: PolicySpec, summary: dict[str, float]) -> None:
    print(f"Agent A: {policy_a.label}")
    print(f"Agent B: {policy_b.label}")
    print(f"games: {int(summary['games'])}")
    print(
        "A win: {a_win_rate:.3f} | B win: {b_win_rate:.3f} | draw: {draw_rate:.3f}".format(
            **summary
        )
    )
    print(
        "mean score A: {mean_a_score:.2f} | B: {mean_b_score:.2f} | margin: {mean_margin:.2f}".format(
            **summary
        )
    )

    category_keys = [key for key in summary if key.startswith("a_") and key[2:] in c.CATEGORY_NAMES]
    if category_keys:
        print("\nPer-category means")
        print(f"{'Category':20} {'A':>8} {'B':>8}")
        for name in c.CATEGORY_NAMES:
            print(f"{name:20} {summary[f'a_{name}']:8.2f} {summary[f'b_{name}']:8.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Yahtzee policies.")
    parser.add_argument("--agent-a", choices=["random", "heuristic", "greedy", "mcts"], default="random")
    parser.add_argument("--agent-b", choices=["random", "heuristic", "greedy", "mcts"], default="random")
    parser.add_argument("--checkpoint-a", type=str, default=None)
    parser.add_argument("--checkpoint-b", type=str, default=None)
    parser.add_argument("--sims-a", type=int, default=32)
    parser.add_argument("--sims-b", type=int, default=32)
    parser.add_argument("--num-games", "--batch-size", dest="num_games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-category", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_a = load_policy(args.agent_a, args.checkpoint_a, args.sims_a)
    policy_b = load_policy(args.agent_b, args.checkpoint_b, args.sims_b)
    results = evaluate_matchup(policy_a, policy_b, args.num_games, args.seed)
    print_summary(policy_a, policy_b, summarize(results, args.per_category))


if __name__ == "__main__":
    main()
