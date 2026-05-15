"""Human-vs-agent command line Yahtzee."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from yahtzee_rl import constants as c
from yahtzee_rl.env import EnvState, legal_action_mask, observation, reset, step
from yahtzee_rl.mcts import search_policy
from yahtzee_rl.model import YahtzeeActorCritic, masked_logits
from yahtzee_rl.scoring import score_categories, total_score
from yahtzee_rl.train import TrainConfig, create_train_state, load_checkpoint


def _scalar(x) -> int:
    return int(jax.device_get(x))


def _state_for_player(state: EnvState, player: int) -> EnvState:
    return state._replace(active_player=jnp.array([player], dtype=jnp.int32))


def _format_score(value: int) -> str:
    return "--" if value < 0 else str(value)


def print_scoreboard(state: EnvState) -> None:
    scorecards = jax.device_get(state.scorecards[0])
    totals = jax.device_get(total_score(state.scorecards)[0])
    print("\nScorecard")
    print("-" * 48)
    print(f"{'Category':20} {'You':>8} {'Agent':>8}")
    for idx, name in enumerate(c.CATEGORY_NAMES):
        print(f"{name:20} {_format_score(int(scorecards[0, idx])):>8} {_format_score(int(scorecards[1, idx])):>8}")
    print("-" * 48)
    print(f"{'Total':20} {int(totals[0]):>8} {int(totals[1]):>8}")


def hold_mask_label(action: int, dice: list[int]) -> str:
    if action == 0:
        return "reroll all dice"
    mask = [(action >> i) & 1 for i in range(c.NUM_DICE)]
    kept = [str(die) for die, keep in zip(dice, mask) if keep]
    return "keep " + " ".join(kept)


def legal_human_actions(state: EnvState) -> list[int]:
    mask = jax.device_get(legal_action_mask(state)[0])
    return [idx for idx, legal in enumerate(mask.tolist()) if legal]


def print_human_options(state: EnvState) -> None:
    dice = jax.device_get(state.dice[0]).tolist()
    scores = jax.device_get(score_categories(state.dice)[0]).tolist()
    legal = legal_human_actions(state)

    if state.rolls_left[0] > 0:
        print("\nReroll actions:")
        for action in legal:
            if action < c.NUM_HOLD_ACTIONS:
                print(f"  h{action:02d}: {hold_mask_label(action, dice)}")

    print("\nScore actions:")
    for action in legal:
        if action >= c.NUM_HOLD_ACTIONS:
            category = action - c.NUM_HOLD_ACTIONS
            print(f"  s{category:02d}: {c.CATEGORY_NAMES[category]} ({scores[category]} pts)")


def parse_human_action(raw: str) -> int | None:
    raw = raw.strip().lower()
    if raw in {"q", "quit", "exit"}:
        raise KeyboardInterrupt
    if not raw:
        return None
    prefix = raw[0]
    rest = raw[1:] if prefix in {"h", "s"} else raw
    if not rest.isdigit():
        return None
    value = int(rest)
    if prefix == "s":
        return c.NUM_HOLD_ACTIONS + value
    if prefix == "h":
        return value
    return value


def ask_human_action(state: EnvState) -> int:
    legal = set(legal_human_actions(state))
    print_human_options(state)
    while True:
        raw = input("\nChoose action (h00-h31, s00-s12, or q): ")
        action = parse_human_action(raw)
        if action in legal:
            return action
        print("That action is not legal here.")


def agent_action(model, params, state: EnvState, key: jax.Array, num_simulations: int, use_mcts: bool) -> int:
    if use_mcts:
        policy = search_policy(model, params, state, key, num_simulations=num_simulations)
        return _scalar(policy.action[0])

    logits, _ = model.apply(params, observation(state))
    logits = masked_logits(logits, legal_action_mask(state))
    return _scalar(jnp.argmax(logits, axis=-1)[0])


def describe_action(action: int, dice: list[int]) -> str:
    if action < c.NUM_HOLD_ACTIONS:
        return hold_mask_label(action, dice)
    category = action - c.NUM_HOLD_ACTIONS
    return f"score {c.CATEGORY_NAMES[category]}"


def load_or_init_agent(checkpoint: str | None, hidden_dim: int):
    if checkpoint:
        state, model, step_idx = load_checkpoint(checkpoint)
        print(f"Loaded checkpoint step {step_idx} from {Path(checkpoint).expanduser()}")
        return state.params, model

    config = TrainConfig(batch_size=1, hidden_dim=hidden_dim)
    state, model, _ = create_train_state(config)
    print("No checkpoint supplied; using an untrained network.")
    return state.params, model


def play(args) -> None:
    params, model = load_or_init_agent(args.checkpoint, args.hidden_dim)
    key = jax.random.PRNGKey(args.seed)
    key, reset_key = jax.random.split(key)
    state = reset(reset_key, batch_size=1)

    try:
        while not bool(jax.device_get(state.done[0])):
            active = _scalar(state.active_player[0])
            print_scoreboard(state)
            dice = jax.device_get(state.dice[0]).tolist()
            rolls_left = _scalar(state.rolls_left[0])
            print(f"\nCurrent dice: {dice} | rolls left: {rolls_left}")

            if active == 0:
                action = ask_human_action(state)
            else:
                key, agent_key = jax.random.split(key)
                agent_state = _state_for_player(state, 1)
                action = agent_action(
                    model,
                    {"params": params},
                    agent_state,
                    agent_key,
                    args.num_simulations,
                    not args.no_mcts,
                )
                print(f"\nAgent chooses: {describe_action(action, dice)}")

            key, step_key = jax.random.split(key)
            state, reward = step(state, jnp.array([action]), step_key)
            if action >= c.NUM_HOLD_ACTIONS:
                print(f"Scored. Reward from actor perspective: {float(reward[0]):.0f}")

    except KeyboardInterrupt:
        print("\nGame ended.")
        return

    print_scoreboard(state)
    totals = jax.device_get(total_score(state.scorecards)[0])
    if totals[0] > totals[1]:
        print("\nYou win.")
    elif totals[1] > totals[0]:
        print("\nAgent wins.")
    else:
        print("\nDraw.")


def parse_args():
    parser = argparse.ArgumentParser(description="Play Yahtzee against the trained agent.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir or step path.")
    parser.add_argument("--num-simulations", type=int, default=32, help="MCTS simulations for agent turns.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dim for untrained agent fallback.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mcts", action="store_true", help="Use greedy network logits instead of MCTS.")
    return parser.parse_args()


def main() -> None:
    play(parse_args())


if __name__ == "__main__":
    main()
