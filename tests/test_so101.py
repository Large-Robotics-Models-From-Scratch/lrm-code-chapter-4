import numpy as np
import pytest

from ch04.so101 import (
    SO101_ACTION_NAMES,
    export_action_chunk,
    load_action_chunk,
    main,
    replay_action_chunk,
)


def test_chunk_export_round_trip_and_dry_run(tmp_path, capsys):
    actions = np.linspace(-1, 1, 16 * 6, dtype=np.float32).reshape(16, 6)
    path = export_action_chunk(
        tmp_path / "chunk.npz",
        actions,
        action_min=np.full(6, -2.0),
        action_max=np.full(6, 2.0),
    )
    payload = load_action_chunk(path)
    np.testing.assert_allclose(payload["actions"], actions)
    assert payload["action_names"] == SO101_ACTION_NAMES
    assert main([str(path), "--port", "/dev/null"]) == 0
    assert "dry run only" in capsys.readouterr().out


def test_chunk_export_rejects_out_of_range_commands(tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        export_action_chunk(
            tmp_path / "bad.npz",
            np.ones((16, 6), dtype=np.float32),
            action_min=np.full(6, -0.5),
            action_max=np.full(6, 0.5),
        )


def test_replay_sends_named_actions(monkeypatch):
    sent = []

    class Robot:
        def send_action(self, action):
            sent.append(action)

    monkeypatch.setattr("time.sleep", lambda _: None)
    replay_action_chunk(Robot(), np.zeros((16, 6), dtype=np.float32))
    assert len(sent) == 16
    assert tuple(sent[0]) == SO101_ACTION_NAMES
