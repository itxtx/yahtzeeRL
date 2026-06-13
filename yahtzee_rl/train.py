"""Training entrypoint for Yahtzee self-play."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import rlax
from flax.training import train_state
from tqdm import trange

import yahtzee_rl.constants as c
from yahtzee_rl.env import observation, reset
from yahtzee_rl.model import YahtzeeActorCritic, legal_mask_from_obs, masked_logits
from yahtzee_rl.rewards import REWARD_MODES, WIN_LOSS_MARGIN
from yahtzee_rl.self_play import Trajectory, generate_self_play, trajectory_observation


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    steps: int = 1000
    batch_size: int = 64
    hidden_dim: int = 256
    num_simulations: int = 32
    learning_rate: float = 3e-4
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 100
    log_every: int = 10
    reward_mode: str = WIN_LOSS_MARGIN
    margin_weight: float = 0.25
    margin_scale: float = 50.0
    # Replay settings: each update generates one batch of games, then takes
    # several SGD steps on minibatches sampled from a buffer of recent frames,
    # amortizing the (expensive) search-based generation cost.
    buffer_size: int = 100_000
    minibatches_per_update: int = 4
    minibatch_size: int = 1024
    # Value target: outcome_weight * terminal_outcome
    #             + (1 - outcome_weight) * search_root_value.
    value_target_outcome_weight: float = 0.5
    # Optional periodic high-quality search teacher. When enabled, every
    # teacher_every-th global update generates self-play with the teacher
    # batch/simulation settings, then can take a different number of SGD steps.
    teacher_every: int = 0
    teacher_num_simulations: int | None = None
    teacher_batch_size: int | None = None
    teacher_minibatches_per_update: int | None = None
    teacher_minibatch_size: int | None = None


FRAME_KEYS = (
    "own_filled",
    "own_scores",
    "opp_filled",
    "opp_scores",
    "upper_own",
    "upper_opp",
    "dice_counts",
    "num_unknown",
    "rolls_left",
    "opponent_to_move",
    "action_weights",
    "value_target",
)


class ReplayBuffer:
    """Fixed-capacity host-side ring buffer of training frames."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: dict[str, np.ndarray] | None = None
        self.size = 0
        self.position = 0

    def add(self, frames: dict[str, np.ndarray]) -> None:
        count = frames[FRAME_KEYS[0]].shape[0]
        if self.storage is None:
            self.storage = {
                key: np.zeros((self.capacity,) + value.shape[1:], dtype=value.dtype)
                for key, value in frames.items()
            }
        indices = (self.position + np.arange(count)) % self.capacity
        for key, value in frames.items():
            self.storage[key][indices] = value
        self.position = int((self.position + count) % self.capacity)
        self.size = int(min(self.size + count, self.capacity))

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, self.size, size=batch_size)
        return {key: value[indices] for key, value in self.storage.items()}


class AgentState(train_state.TrainState):
    pass


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def teacher_enabled(config: TrainConfig) -> bool:
    return config.teacher_every > 0


def validate_config(config: TrainConfig) -> None:
    _positive("batch_size", config.batch_size)
    _positive("num_simulations", config.num_simulations)
    _positive("minibatches_per_update", config.minibatches_per_update)
    _positive("minibatch_size", config.minibatch_size)
    if config.teacher_every < 0:
        raise ValueError(f"teacher_every must be non-negative, got {config.teacher_every}")

    teacher_overrides = (
        config.teacher_num_simulations,
        config.teacher_batch_size,
        config.teacher_minibatches_per_update,
        config.teacher_minibatch_size,
    )
    has_teacher_override = any(value is not None for value in teacher_overrides)
    if config.teacher_every == 0:
        if has_teacher_override:
            raise ValueError("Teacher schedule options require --teacher-every > 0.")
        return
    if not has_teacher_override:
        raise ValueError("--teacher-every requires at least one teacher override.")

    if config.teacher_num_simulations is not None:
        _positive("teacher_num_simulations", config.teacher_num_simulations)
    if config.teacher_batch_size is not None:
        _positive("teacher_batch_size", config.teacher_batch_size)
    if config.teacher_minibatches_per_update is not None:
        _positive(
            "teacher_minibatches_per_update",
            config.teacher_minibatches_per_update,
        )
    if config.teacher_minibatch_size is not None:
        _positive("teacher_minibatch_size", config.teacher_minibatch_size)


def teacher_play_config(config: TrainConfig) -> TrainConfig:
    return replace(
        config,
        batch_size=(
            config.teacher_batch_size
            if config.teacher_batch_size is not None
            else config.batch_size
        ),
        num_simulations=(
            config.teacher_num_simulations
            if config.teacher_num_simulations is not None
            else config.num_simulations
        ),
    )


def update_minibatch_settings(config: TrainConfig, use_teacher: bool) -> tuple[int, int]:
    if not use_teacher:
        return config.minibatches_per_update, config.minibatch_size
    return (
        config.teacher_minibatches_per_update
        if config.teacher_minibatches_per_update is not None
        else config.minibatches_per_update,
        config.teacher_minibatch_size
        if config.teacher_minibatch_size is not None
        else config.minibatch_size,
    )


def is_teacher_step(config: TrainConfig, step: int) -> bool:
    return teacher_enabled(config) and step % config.teacher_every == 0


def create_train_state(config: TrainConfig) -> tuple[AgentState, YahtzeeActorCritic, jax.Array]:
    key = jax.random.PRNGKey(config.seed)
    key, init_key, env_key = jax.random.split(key, 3)
    model = YahtzeeActorCritic(hidden_dims=(config.hidden_dim, config.hidden_dim))
    dummy_state = reset(env_key, config.batch_size)
    params = model.init(init_key, observation(dummy_state))["params"]
    tx = optax.adam(config.learning_rate)
    state = AgentState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model, key


def frames_from_trajectory(trajectory: Trajectory, config: TrainConfig) -> dict[str, np.ndarray]:
    """Flatten a trajectory into valid frames with blended value targets."""
    obs = trajectory_observation(trajectory)
    valid = np.asarray(jax.device_get(trajectory.valid.reshape((-1,))))
    outcome_weight = config.value_target_outcome_weight
    value_target = (
        outcome_weight * trajectory.returns.reshape((-1,))
        + (1.0 - outcome_weight) * trajectory.search_values.reshape((-1,))
    )
    frames = {key: np.asarray(jax.device_get(obs[key])) for key in obs if key != "active_player"}
    frames["action_weights"] = np.asarray(
        jax.device_get(trajectory.action_weights.reshape((-1, c.NUM_ACTIONS)))
    )
    frames["value_target"] = np.asarray(jax.device_get(value_target))
    return {key: value[valid] for key, value in frames.items()}


def make_generate_fn(model: YahtzeeActorCritic, config: TrainConfig):
    def generate(params, rng_key):
        return generate_self_play(
            model,
            {"params": params},
            rng_key,
            batch_size=config.batch_size,
            num_simulations=config.num_simulations,
            reward_mode=config.reward_mode,
            margin_weight=config.margin_weight,
            margin_scale=config.margin_scale,
        )

    return jax.jit(generate)


def make_train_step(model: YahtzeeActorCritic, config: TrainConfig):
    def loss_fn(params, batch):
        obs = {
            "own_filled": batch["own_filled"],
            "own_scores": batch["own_scores"],
            "opp_filled": batch["opp_filled"],
            "opp_scores": batch["opp_scores"],
            "upper_own": batch["upper_own"],
            "upper_opp": batch["upper_opp"],
            "dice_counts": batch["dice_counts"],
            "num_unknown": batch["num_unknown"],
            "rolls_left": batch["rolls_left"],
            "opponent_to_move": batch["opponent_to_move"],
        }
        logits, values = model.apply({"params": params}, obs)
        logits = masked_logits(logits, legal_mask_from_obs(obs))

        log_probs = jax.nn.log_softmax(logits)
        policy_loss = jnp.mean(-jnp.sum(batch["action_weights"] * log_probs, axis=-1))
        value_loss = jnp.mean(rlax.l2_loss(values - batch["value_target"]))
        entropy = jnp.mean(-jnp.sum(jax.nn.softmax(logits) * log_probs, axis=-1))
        total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        metrics = {
            "loss": total,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }
        return total, metrics

    def train_step(state: AgentState, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch
        )
        return state.apply_gradients(grads=grads), metrics

    return jax.jit(train_step)


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
    if not config_path.exists():
        # Silently assuming default hyperparameters (e.g. hidden_dim) would
        # fail later with opaque parameter-shape errors; fail loudly instead.
        raise FileNotFoundError(
            f"No config.json next to checkpoint {checkpoint_path}; cannot "
            "reconstruct the model architecture used at training time."
        )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = asdict(TrainConfig())
    defaults.update(raw_config)
    return TrainConfig(**defaults)


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


def train(config: TrainConfig, resume_from: str | None = None) -> AgentState:
    validate_config(config)
    state, model, key = create_train_state(config)
    start_step = 0
    if resume_from is not None:
        checkpoint_config = load_checkpoint_config(resume_from)
        if checkpoint_config.hidden_dim != config.hidden_dim:
            raise ValueError(
                f"Checkpoint was trained with hidden_dim="
                f"{checkpoint_config.hidden_dim} but this run requests "
                f"hidden_dim={config.hidden_dim}; the architectures must match "
                "to resume."
            )
        state, model, start_step = load_checkpoint(resume_from, config)
        # Fold the restored step into the rng stream so a resumed run does not
        # replay the same self-play games as a fresh run with the same seed.
        key = jax.random.fold_in(key, start_step)
        print(f"Resumed params and optimizer state from step {start_step}")

    generate = make_generate_fn(model, config)
    teacher_config = teacher_play_config(config) if teacher_enabled(config) else None
    teacher_generate = (
        make_generate_fn(model, teacher_config) if teacher_config is not None else None
    )
    train_step = make_train_step(model, config)
    buffer = ReplayBuffer(config.buffer_size)
    sample_rng = np.random.default_rng(config.seed + start_step)
    checkpoint_dir = Path(config.checkpoint_dir)

    print(f"JAX devices: {jax.devices()}")
    print(
        f"Training {config.steps} updates with batch={config.batch_size}, "
        f"sims={config.num_simulations}, "
        f"minibatches={config.minibatches_per_update}x{config.minibatch_size}"
        + (f", resuming at global step {start_step}" if start_step else "")
    )
    if teacher_config is not None:
        teacher_minibatches, teacher_minibatch_size = update_minibatch_settings(
            config, use_teacher=True
        )
        print(
            f"Teacher update every {config.teacher_every} global steps with "
            f"batch={teacher_config.batch_size}, "
            f"sims={teacher_config.num_simulations}, "
            f"minibatches={teacher_minibatches}x{teacher_minibatch_size}"
        )

    last_log = time.time()
    for step_idx in trange(start_step + 1, start_step + config.steps + 1):
        key, play_key = jax.random.split(key)
        use_teacher = is_teacher_step(config, step_idx)
        play_config = (
            teacher_config if use_teacher and teacher_config is not None else config
        )
        generate_fn = (
            teacher_generate
            if use_teacher and teacher_generate is not None
            else generate
        )
        trajectory, play_metrics = generate_fn(state.params, play_key)
        buffer.add(frames_from_trajectory(trajectory, config))

        metrics = {}
        minibatches, minibatch_size = update_minibatch_settings(config, use_teacher)
        for _ in range(minibatches):
            batch = buffer.sample(minibatch_size, sample_rng)
            state, metrics = train_step(state, batch)

        if step_idx % config.log_every == 0 or step_idx == start_step + 1:
            metrics = metrics | play_metrics | {"buffer_size": buffer.size}
            ready_metrics = jax.tree_util.tree_map(lambda x: float(jax.device_get(x)), metrics)
            now = time.time()
            steps_per_sec = config.log_every / max(now - last_log, 1e-6)
            last_log = now
            print(
                "step={step} mode={mode} gen={play_batch}x{play_sims} "
                "sgd={sgd_minibatches}x{sgd_minibatch_size} "
                "loss={loss:.4f} policy={policy_loss:.4f} "
                "value={value_loss:.4f} entropy={entropy:.3f} "
                "p0={mean_player0_score:.1f} p1={mean_player1_score:.1f} "
                "p0w={player0_win_rate:.2f} p1w={player1_win_rate:.2f} "
                "draw={draw_rate:.2f} margin={mean_score_margin_player0:.1f} "
                "abs_margin={mean_abs_score_margin:.1f} shaped={mean_shaped_return:.3f} "
                "buffer={buffer_size:.0f} ups={ups:.2f}".format(
                    step=step_idx,
                    mode="teacher" if use_teacher else "base",
                    play_batch=play_config.batch_size,
                    play_sims=play_config.num_simulations,
                    sgd_minibatches=minibatches,
                    sgd_minibatch_size=minibatch_size,
                    ups=steps_per_sec,
                    **ready_metrics,
                )
            )

        if step_idx % config.checkpoint_every == 0 or step_idx == start_step + config.steps:
            save_checkpoint(checkpoint_dir, state, config, step_idx)

    return state


def parse_args() -> tuple[TrainConfig, str | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint dir (or step path) to resume from. Restores params and "
            "optimizer state, continues global step numbering, and runs --steps "
            "additional updates. Other flags (e.g. --num-simulations) may differ "
            "from the original run; --hidden-dim must match."
        ),
    )
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
    parser.add_argument("--buffer-size", type=int, default=TrainConfig.buffer_size)
    parser.add_argument(
        "--minibatches-per-update", type=int, default=TrainConfig.minibatches_per_update
    )
    parser.add_argument("--minibatch-size", type=int, default=TrainConfig.minibatch_size)
    parser.add_argument(
        "--value-target-outcome-weight",
        type=float,
        default=TrainConfig.value_target_outcome_weight,
    )
    parser.add_argument(
        "--teacher-every",
        type=int,
        default=TrainConfig.teacher_every,
        help=(
            "Enable periodic teacher updates every N global steps. Teacher "
            "updates can use higher simulations, a smaller batch, and different "
            "SGD minibatch settings."
        ),
    )
    parser.add_argument(
        "--teacher-num-simulations",
        type=int,
        default=TrainConfig.teacher_num_simulations,
    )
    parser.add_argument(
        "--teacher-batch-size",
        type=int,
        default=TrainConfig.teacher_batch_size,
    )
    parser.add_argument(
        "--teacher-minibatches-per-update",
        type=int,
        default=TrainConfig.teacher_minibatches_per_update,
    )
    parser.add_argument(
        "--teacher-minibatch-size",
        type=int,
        default=TrainConfig.teacher_minibatch_size,
    )
    args = parser.parse_args()
    arg_dict = vars(args)
    resume_from = arg_dict.pop("resume")
    return TrainConfig(**arg_dict), resume_from


def main() -> None:
    config, resume_from = parse_args()
    train(config, resume_from=resume_from)


if __name__ == "__main__":
    main()
