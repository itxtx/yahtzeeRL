"""Flax policy/value model for Yahtzee."""

import jax
import jax.numpy as jnp
from flax import linen as nn

import yahtzee_rl.constants as c


def flatten_observation(obs: dict[str, jax.Array]) -> jax.Array:
    """Flatten the active-player observation into a dense feature vector."""
    dice_one_hot = jax.nn.one_hot(obs["dice"] - 1, c.NUM_FACES).reshape(
        obs["dice"].shape[0], c.NUM_DICE * c.NUM_FACES
    )
    rolls_one_hot = jax.nn.one_hot(obs["rolls_left"], c.MAX_ROLLS_LEFT + 1)
    return jnp.concatenate(
        [
            obs["own_filled"],
            obs["own_scores"],
            obs["opp_filled"],
            obs["opp_scores"],
            dice_one_hot,
            rolls_one_hot,
        ],
        axis=-1,
    )


class YahtzeeActorCritic(nn.Module):
    """Shared-backbone policy/value network."""

    hidden_dims: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, obs: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x = flatten_observation(obs)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        policy_logits = nn.Dense(c.NUM_ACTIONS)(x)
        value = nn.Dense(1)(x)
        return policy_logits, jnp.tanh(value[..., 0])


def masked_logits(logits: jax.Array, legal_mask: jax.Array) -> jax.Array:
    return jnp.where(legal_mask, logits, jnp.finfo(logits.dtype).min)
