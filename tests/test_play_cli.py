from yahtzee_rl.play_cli import compact_hold_action_labels


def test_compact_hold_action_labels_collapses_duplicate_dice_values():
    labels = compact_hold_action_labels(list(range(32)), [3, 4, 5, 5, 6])

    assert "h04 keep 5 (same as h08)" in labels
    assert "h20 keep 5 6 (same as h24)" in labels
    assert not any(label.startswith("h08 keep 5") for label in labels)
    assert len(labels) == 24
