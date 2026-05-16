import jax

from yahtzee_rl.evaluate import evaluate_matchup, load_policy, summarize


def test_random_vs_random_evaluation_smoke():
    policy_a = load_policy("random", None, 1)
    policy_b = load_policy("random", None, 1)

    results = evaluate_matchup(policy_a, policy_b, num_games=4, seed=0)
    summary = summarize(results)

    assert summary["games"] == 4
    assert 0.0 <= summary["a_win_rate"] <= 1.0
    assert 0.0 <= summary["b_win_rate"] <= 1.0
    assert 0.0 <= summary["draw_rate"] <= 1.0


def test_cli_agent_debug_formatter_without_mcts():
    from yahtzee_rl.env import reset
    from yahtzee_rl.play_cli import agent_debug_lines
    from yahtzee_rl.train import TrainConfig, create_train_state

    train_state, model, _ = create_train_state(TrainConfig(batch_size=1, hidden_dim=8))
    state = reset(jax.random.PRNGKey(0), batch_size=1)
    action, lines = agent_debug_lines(
        model,
        {"params": train_state.params},
        state,
        jax.random.PRNGKey(1),
        num_simulations=1,
        use_mcts=False,
        top_k=3,
    )

    assert 0 <= action < 45
    assert any("Agent value estimate" in line for line in lines)
    assert sum(line.strip().startswith(("1.", "2.", "3.")) for line in lines) == 3
