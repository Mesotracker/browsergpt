from __future__ import annotations

import math

import pytest

from chainforge.config import TrainConfig
from chainforge.trainer import TokenData, TrainingCancelled, train_model, pick_device
from tests.helpers import tiny_model, tiny_tokenizer, write_token_file


def _setup(tmp_path):
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, ctx=32)
    tokens = write_token_file(tmp_path / "tokens.bin", tokenizer, repeats=8)
    data = TokenData(tmp_path / "tokens.bin", val_frac=0.05)
    config = TrainConfig(
        steps=6,
        batch_size=2,
        lr=3e-3,
        eval_interval=3,
        eval_batches=1,
        log_every=1,
        seed=7,
    )
    return tokenizer, model, tokens, data, config


def test_training_runs_and_reports(tmp_path):
    _tokenizer, model, tokens, data, config = _setup(tmp_path)
    events = []
    report = train_model(model, data, config, device="cpu", on_event=events.append)
    assert tokens >= 512
    assert report.steps == 6
    assert len(report.losses) == 6
    assert all(math.isfinite(loss) for loss in report.losses)
    assert report.val_losses
    assert math.isfinite(report.final_loss)
    assert report.tokens_seen == 6 * 2 * 32
    assert any(event["type"] == "step" for event in events)
    assert any(event["type"] == "eval" for event in events)


def test_validation_split_comes_from_the_tail(tmp_path):
    tokenizer = tiny_tokenizer()
    write_token_file(tmp_path / "tokens.bin", tokenizer, repeats=4)
    data = TokenData(tmp_path / "tokens.bin", val_frac=0.1)
    assert data.train_len + data.val_len == data.total_tokens
    assert data.val_len == max(1, int(data.total_tokens * 0.1))


def test_cancellation_stops_before_a_step(tmp_path):
    _tokenizer, model, _tokens, data, config = _setup(tmp_path)
    with pytest.raises(TrainingCancelled):
        train_model(
            model,
            data,
            config,
            device="cpu",
            should_stop=lambda: True,
        )


def test_device_resolution_is_a_string():
    assert isinstance(pick_device("cpu"), str)
    assert pick_device("cpu") == "cpu"
