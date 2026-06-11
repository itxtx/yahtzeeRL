import jax
import numpy as np
import pytest

from yahtzee_rl.env import observation, reset
from yahtzee_rl.model import YahtzeeActorCritic
from yahtzee_rl.self_play import generate_self_play
from yahtzee_rl.train import (
    ReplayBuffer,
    TrainConfig,
    frames_from_trajectory,
    load_checkpoint_config,
    make_train_step,
)


def test_replay_buffer_ring_semantics():
    buffer = ReplayBuffer(capacity=10)
    frames = {"own_filled": np.arange(8, dtype=np.float32).reshape(8, 1)}
    buffer.add(frames)
    assert buffer.size == 8
    buffer.add(frames)  # wraps: 16 > 10
    assert buffer.size == 10
    batch = buffer.sample(4, np.random.default_rng(0))
    assert batch["own_filled"].shape == (4, 1)


def test_frames_have_blended_value_targets_and_train_step_runs():
    config = TrainConfig(
        batch_size=1,
        hidden_dim=8,
        num_simulations=2,
        minibatch_size=8,
        value_target_outcome_weight=0.5,
    )
    model = YahtzeeActorCritic(hidden_dims=(8, 8))
    state = reset(jax.random.PRNGKey(0), 1)
    params = model.init(jax.random.PRNGKey(1), observation(state))
    trajectory, _ = generate_self_play(
        model, params, jax.random.PRNGKey(2), batch_size=1, num_simulations=2
    )

    frames = frames_from_trajectory(trajectory, config)
    valid = np.asarray(trajectory.valid.reshape(-1))
    returns = np.asarray(trajectory.returns.reshape(-1))[valid]
    search_values = np.asarray(trajectory.search_values.reshape(-1))[valid]
    np.testing.assert_allclose(
        frames["value_target"], 0.5 * returns + 0.5 * search_values, rtol=1e-5
    )
    assert frames["action_weights"].shape[0] == valid.sum()

    buffer = ReplayBuffer(capacity=128)
    buffer.add(frames)
    batch = buffer.sample(config.minibatch_size, np.random.default_rng(0))
    train_step = make_train_step(model, config)
    from yahtzee_rl.train import create_train_state

    agent_state, _, _ = create_train_state(config)
    new_state, metrics = train_step(agent_state, batch)
    assert np.isfinite(float(metrics["loss"]))
    assert int(new_state.step) == 1


def test_missing_checkpoint_config_raises(tmp_path):
    step_dir = tmp_path / "step_000100"
    step_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_checkpoint_config(step_dir)
