from __future__ import annotations

from chainforge.generate import chat_stream, fit_prompt
from tests.helpers import tiny_model, tiny_tokenizer


def test_prompt_is_trimmed_into_the_context_window():
    tokenizer = tiny_tokenizer()
    messages = [
        {"role": "user", "content": "An earlier question about token budgets. " * 20},
        {"role": "assistant", "content": "An earlier answer. " * 20},
        {"role": "user", "content": "Now explain attention in detail. " * 40},
    ]
    ids, budget = fit_prompt(
        tokenizer,
        messages,
        48,
        reasoning=True,
        think_tokens=12,
        answer_tokens=16,
    )
    assert len(ids) <= 48
    assert budget.truncated_tokens > 0
    assert budget.context == 48
    assert budget.think_budget == 12
    assert budget.answer_budget >= 2


def test_chat_stream_finishes_and_respects_budgets():
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, ctx=48)
    events = list(
        chat_stream(
            model,
            tokenizer,
            [{"role": "user", "content": "Hello there."}],
            reasoning=True,
            think_tokens=8,
            answer_tokens=8,
            temperature=0.0,
            top_k=1,
        )
    )
    assert events[0]["type"] == "start"
    assert events[-1]["type"] == "done"
    budget = events[-1]["budget"]
    assert budget["think_used"] <= budget["think_budget"]
    assert budget["answer_used"] <= budget["answer_budget"]
    assert any(event["type"] == "reasoning_end" for event in events)


def test_reasoning_can_be_disabled():
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, ctx=48)
    events = list(
        chat_stream(
            model,
            tokenizer,
            [{"role": "user", "content": "Hello."}],
            reasoning=False,
            think_tokens=8,
            answer_tokens=8,
            temperature=0.0,
        )
    )
    budget = events[-1]["budget"]
    assert budget["reasoning_stop"] == "off"
    assert budget["think_used"] == 0
    assert all(event["type"] != "reasoning" for event in events)
