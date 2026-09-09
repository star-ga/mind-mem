"""Model cards must not replace missing run evidence with invented metrics."""

import json

import pytest

from train import build_model_card


@pytest.mark.parametrize("state", (None, "{", "{}", '{"log_history":[]}'))
def test_missing_training_metrics_remain_unavailable(tmp_path, monkeypatch, state):
    monkeypatch.setenv("MM_TRAIN_ROOT", str(tmp_path))
    if state is not None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "trainer_state.json").write_text(state, encoding="utf-8")

    assert build_model_card._load_train_metrics() == {
        "final_loss": "_not available_",
        "train_loss_mean": "_not available_",
        "token_accuracy": "_not available_",
    }


def test_partial_training_evidence_preserves_only_recorded_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("MM_TRAIN_ROOT", str(tmp_path))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    path = adapter / "trainer_state.json"
    state = {"log_history": [{"loss": 0.5}, {"loss": 0.125, "mean_token_accuracy": 0.875}]}
    path.write_text(json.dumps(state), encoding="utf-8")

    assert build_model_card._load_train_metrics() == {
        "final_loss": "0.125",
        "train_loss_mean": "_not available_",
        "token_accuracy": "87.5%",
    }

    state["train_loss"] = 0.25
    path.write_text(json.dumps(state), encoding="utf-8")
    assert build_model_card._load_train_metrics()["train_loss_mean"] == "0.25"
