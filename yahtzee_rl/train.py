"""Training entrypoint for Yahtzee self-play."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import rlax
from flax.training import train_state
from tqdm import trange

import yahtzee_rl.constants as c
from yahtzee_rl.env import observation, reset
from yahtzee_rl.model import YahtzeeActorCritic, legal_mask_from_obs, masked_logits
from yahtzee_rl.rewards import REWARD_MODES, WIN_LOSS_MARGIN
from yahtzee_rl.self_play import generate_self_play, trajectory_observation


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    steps: int = 1000
    batch_size: int = 64
    hidden_dim: int = 256
    num_simulations: int = 32
    learning_rate: float = 3e-4
    value_coef: float = 1.0
    entropy_coef: float = 1e-3
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 100
    log_every: int = 10
    reward_mode: str = WIN_LOSS_MARGIN
    margin_weight: float = 0.25
    margin_scale: float = 50.0


class AgentState(train_state.TrainState):
    pass


def create_train_state(config: TrainConfig) -> tuple[AgentState, YahtzeeActorCritic, jax.Array]:
    key = jax.random.PRNGKey(config.seed)
    key, init_key, env_key = jax.random.split(key, 3)
    model = YahtzeeActorCritic(hidden_dims=(config.hidden_dim, config.hidden_dim))
    dummy_state = reset(env_key, config.batch_size)
    params = model.init(init_key, observation(dummy_state))["params"]
    tx = optax.adam(config.learning_rate)
    state = AgentState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model, key


def make_update_fn(model: YahtzeeActorCritic, config: TrainConfig):
    def loss_fn(params, trajectory):
        obs = trajectory_observation(trajectory)
        logits, values = model.apply({"params": params}, obs)
        logits = masked_logits(logits, legal_mask_from_obs(obs))

        action_weights = trajectory.action_weights.reshape((-1, c.NUM_ACTIONS))
        returns = trajectory.returns.reshape((-1,))
        valid = trajectory.valid.reshape((-1,)).astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(valid), 1.0)

        log_probs = jax.nn.log_softmax(logits)
        policy_loss = -jnp.sum(action_weights * log_probs, axis=-1)
        value_loss = rlax.l2_loss(values - returns)
        entropy = -jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1)

        policy_loss = jnp.sum(policy_loss * valid) / denom
        value_loss = jnp.sum(value_loss * valid) / denom
        entropy = jnp.sum(entropy * valid) / denom
        total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        metrics = {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
        return total, metrics

    def update_step(state: AgentState, rng_key: jax.Array):
        play_key = rng_key
        trajectory, play_metrics = generate_self_play(
            model,
            {"params": state.params},
            play_key,
            batch_size=config.batch_size,
            num_simulations=config.num_simulations,
            reward_mode=config.reward_mode,
            margin_weight=config.margin_weight,
            margin_scale=config.margin_scale,
        )
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, trajectory
        )
        new_state = state.apply_gradients(grads=grads)
        metrics = metrics | play_metrics | {"loss": loss}
        return new_state, metrics

    return jax.jit(update_step)


def save_checkpoint(path: Path, state: AgentState, config: TrainConfig, step: int) -> None:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    payload = {"params": state.params, "opt_state": state.opt_state, "step": step}
    try:
        checkpointer.save(path / f"step_{step:06d}", payload, force=True)
        # Block until the async save fully finalizes so the checkpoint is
        # complete before the process can exit (otherwise orbax background
        # threads can fail during interpreter shutdown).
        checkpointer.wait_until_finished()
    finally:
        checkpointer.close()
    (path / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


_STEP_DIR_RE = re.compile(r"^step_\d+$")


def latest_checkpoint(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    # Only consider finalized checkpoint dirs (step_<digits>), skipping orbax
    # ".orbax-checkpoint-tmp" temp dirs left behind by interrupted saves.
    # Sort by the numeric step (not lexicographically) so step_1000000 is
    # correctly ordered after step_999999.
    candidates = sorted(
        (p for p in path.glob("step_*") if _STEP_DIR_RE.match(p.name)),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {path}")
    return candidates[-1]


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    return latest_checkpoint(path) if path.is_dir() and not path.name.startswith("step_") else path


def load_checkpoint_config(path: str | Path) -> TrainConfig:
    checkpoint_path = resolve_checkpoint_path(path)
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        defaults = asdict(TrainConfig())
        defaults.update(raw_config)
        return TrainConfig(**defaults)
    return TrainConfig(batch_size=1)


def load_checkpoint(path: str | Path, config: TrainConfig | None = None) -> tuple[AgentState, YahtzeeActorCritic, int]:
    checkpoint_path = resolve_checkpoint_path(path)
    if config is None:
        config = load_checkpoint_config(checkpoint_path)

    state, model, _ = create_train_state(config)
    checkpointer = ocp.StandardCheckpointer()
    # Provide an abstract target so arrays deserialize onto the local devices
    # regardless of the sharding used when the checkpoint was written. Without a
    # target, restoring a GPU/TPU-saved checkpoint on CPU fails with a
    # "sharding ... Got None" error.
    target = {"params": state.params, "opt_state": state.opt_state, "step": 0}
    restored = checkpointer.restore(checkpoint_path, target=target)
    state = state.replace(params=restored["params"], opt_state=restored["opt_state"])
    return state, model, int(restored["step"])


def train(config: TrainConfig) -> AgentState:
    state, model, key = create_train_state(config)
    update_step = make_update_fn(model, config)
    checkpoint_dir = Path(config.checkpoint_dir)

    print(f"JAX devices: {jax.devices()}")
    print(f"Training {config.steps} updates with batch={config.batch_size}, sims={config.num_simulations}")

    last_log = time.time()
    for step_idx in trange(1, config.steps + 1):
        key, step_key = jax.random.split(key)
        state, metrics = update_step(state, step_key)

        if step_idx % config.log_every == 0 or step_idx == 1:
            ready_metrics = jax.tree_util.tree_map(lambda x: float(jax.device_get(x)), metrics)
            now = time.time()
            steps_per_sec = config.log_every / max(now - last_log, 1e-6)
            last_log = now
            print(
                "step={step} loss={loss:.4f} policy={policy_loss:.4f} "
                "value={value_loss:.4f} entropy={entropy:.3f} "
                "p0={mean_player0_score:.1f} p1={mean_player1_score:.1f} "
                "p0w={player0_win_rate:.2f} p1w={player1_win_rate:.2f} "
                "draw={draw_rate:.2f} margin={mean_score_margin_player0:.1f} "
                "abs_margin={mean_abs_score_margin:.1f} shaped={mean_shaped_return:.3f} "
                "ups={ups:.2f}".format(
                    step=step_idx, ups=steps_per_sec, **ready_metrics
                )
            )

        if step_idx % config.checkpoint_every == 0 or step_idx == config.steps:
            save_checkpoint(checkpoint_dir, state, config, step_idx)

    return state


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--hidden-dim", type=int, default=TrainConfig.hidden_dim)
    parser.add_argument("--num-simulations", type=int, default=TrainConfig.num_simulations)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--checkpoint-dir", type=str, default=TrainConfig.checkpoint_dir)
    parser.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default=TrainConfig.reward_mode)
    parser.add_argument("--margin-weight", type=float, default=TrainConfig.margin_weight)
    parser.add_argument("--margin-scale", type=float, default=TrainConfig.margin_scale)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
