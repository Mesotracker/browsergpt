"""Sampling and the chat protocol with a visible reasoning scratchpad.

The model is *trained* on a simple protocol:

    <|user|> question <|assistant|> <|think|> notes <|endthink|> answer <|end|>

At inference we stream the notes first, then the answer, and account for every
token against a hard context budget. The scratchpad is a draft the model
conditions on; it is not a faithful record of hidden computation, and the UI
says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import ByteBPE


@dataclass
class GenerationBudget:
    context: int
    prompt_tokens: int
    think_budget: int
    answer_budget: int
    think_used: int = 0
    answer_used: int = 0
    truncated_tokens: int = 0
    reasoning_stop: str | None = None
    answer_stop: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _role(m: dict) -> str:
    role = m.get("role", "user")
    if role not in ("user", "assistant"):
        raise ValueError(f"unsupported message role: {role!r}")
    return role


def _content(m: dict) -> str:
    content = str(m.get("content", "")).strip()
    if not content:
        raise ValueError("messages must have non-empty content")
    return content


def fit_prompt(
    tokenizer: ByteBPE,
    messages: Sequence[dict],
    context: int,
    *,
    reasoning: bool = True,
    think_tokens: int = 48,
    answer_tokens: int = 96,
) -> tuple[list[int], GenerationBudget]:
    if not messages:
        raise ValueError("at least one message is required")
    if _role(messages[-1]) != "user":
        raise ValueError("the last message must be from the user")
    if context < 32:
        raise ValueError("context window is too small")

    last = _content(messages[-1])
    history = list(messages[:-1])

    bos_ids = tokenizer.encode("<|bos|>") if "<|bos|>" in tokenizer.specials else []
    suffix_ids = tokenizer.encode(
        "<|assistant|>\n" + ("<|think|>\n" if reasoning else "")
    )

    history_parts: list[str] = []
    for message in history:
        if _role(message) == "user":
            history_parts.append(
                "<|user|>\n" + _content(message) + "\n<|assistant|>\n"
            )
        else:
            history_parts.append(_content(message) + "<|end|>")
    history_ids = tokenizer.encode("".join(history_parts))
    user_ids = tokenizer.encode("<|user|>\n" + last + "\n")

    # Generation can never eat the whole window; the prompt needs room too.
    max_gen = max(2, context - len(suffix_ids) - len(bos_ids) - 8)
    think_budget = 0
    if reasoning:
        think_budget = max(0, min(int(think_tokens), max_gen - 2))
    answer_budget = max(2, min(int(answer_tokens), max_gen - think_budget))
    if think_budget + answer_budget > max_gen:
        answer_budget = max(2, max_gen - think_budget)

    prompt_room = (
        context
        - len(suffix_ids)
        - think_budget
        - answer_budget
        - len(bos_ids)
        - 2
    )

    truncated = 0
    history_room = min(len(history_ids), max(0, prompt_room // 2))
    truncated += len(history_ids) - history_room
    user_room = max(2, prompt_room - history_room)
    if len(user_ids) > user_room:
        truncated += len(user_ids) - user_room
        user_ids = user_ids[-user_room:]

    kept_history = history_ids[len(history_ids) - history_room :] if history_room else []
    ids = bos_ids + kept_history + user_ids + suffix_ids
    if len(ids) + 2 > context:
        keep = max(2, context - 2)
        truncated += len(ids) - keep
        ids = ids[-keep:]

    budget = GenerationBudget(
        context=context,
        prompt_tokens=len(ids),
        think_budget=think_budget,
        answer_budget=answer_budget,
        truncated_tokens=truncated,
    )
    return ids, budget


def generate_ids(
    model: GPT,
    tokenizer: ByteBPE,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    temperature: float = 0.85,
    top_k: int = 40,
    stop_ids: Sequence[int] = (),
    banned_ids: Sequence[int] = (),
    repetition_penalty: float = 1.1,
    seed: int | None = None,
) -> Iterator[int]:
    """Yield sampled token ids one at a time."""
    if max_new_tokens <= 0:
        return
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    stop = set(int(x) for x in stop_ids)
    banned = [int(x) for x in banned_ids]
    model.eval()
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            condition = idx[:, -model.cfg.ctx :]
            logits, _ = model(condition)
            logits = logits[:, -1, :].float()
            if banned:
                logits[:, banned] = -float("inf")
            if repetition_penalty != 1.0 and generated:
                recent = set(idx[0, -64:].tolist())
                for token in recent:
                    score = logits[0, token]
                    if score > 0:
                        logits[0, token] = score / repetition_penalty
                    else:
                        logits[0, token] = score * repetition_penalty
            if temperature <= 0:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                scaled = logits / max(temperature, 1e-5)
                if top_k > 0:
                    k = min(top_k, scaled.size(-1))
                    values, _indices = torch.topk(scaled, k)
                    cutoff = values[:, -1, None]
                    scaled = torch.where(
                        scaled < cutoff,
                        torch.full_like(scaled, -float("inf")),
                        scaled,
                    )
                probs = F.softmax(scaled, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
            idx = torch.cat(
                [idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
            )
            yield next_id
            if next_id in stop:
                break


def _special(tokenizer: ByteBPE, name: str) -> int | None:
    return tokenizer.specials.get(name)


def chat_stream(
    model: GPT,
    tokenizer: ByteBPE,
    messages: Sequence[dict],
    *,
    reasoning: bool = True,
    think_tokens: int = 48,
    answer_tokens: int = 96,
    temperature: float = 0.85,
    top_k: int = 40,
    seed: int | None = None,
) -> Iterator[dict]:
    """Stream reasoning and answer deltas plus a final token budget."""
    ids, budget = fit_prompt(
        tokenizer,
        messages,
        model.cfg.ctx,
        reasoning=reasoning,
        think_tokens=think_tokens,
        answer_tokens=answer_tokens,
    )

    endthink_id = _special(tokenizer, "<|endthink|>")
    end_id = _special(tokenizer, "<|end|>")
    can_reason = reasoning and endthink_id is not None and budget.think_budget > 0

    yield {"type": "start", "budget": budget.to_dict(), "context": model.cfg.ctx}

    if can_reason:
        think_ids: list[int] = []
        think_text = ""
        stopped_on_endthink = False
        banned = [
            i
            for name, i in tokenizer.specials.items()
            if name not in ("<|endthink|>", "<|end|>")
        ]
        stops = [i for i in (endthink_id, end_id) if i is not None]
        for token_id in generate_ids(
            model,
            tokenizer,
            ids,
            budget.think_budget,
            temperature=temperature,
            top_k=top_k,
            stop_ids=stops,
            banned_ids=banned,
            seed=seed,
        ):
            think_ids.append(token_id)
            budget.think_used += 1
            text = tokenizer.decode(think_ids, skip_special=True)
            delta = text[len(think_text) :]
            think_text = text
            if delta:
                yield {"type": "reasoning", "text": delta}
            if token_id == endthink_id:
                stopped_on_endthink = True
                break
            if token_id == end_id:
                break
        if stopped_on_endthink:
            budget.reasoning_stop = "endthink"
        else:
            budget.reasoning_stop = "budget"
            think_ids.append(endthink_id)
        ids = ids + think_ids + tokenizer.encode("\n")
    else:
        budget.reasoning_stop = "off"
        budget.think_budget = 0
        budget.think_used = 0

    yield {"type": "reasoning_end", "tokens": budget.think_used}

    answer_ids: list[int] = []
    answer_text = ""
    banned = [
        i for name, i in tokenizer.specials.items() if name != "<|end|>"
    ]
    stops = [end_id] if end_id is not None else []
    for token_id in generate_ids(
        model,
        tokenizer,
        ids,
        budget.answer_budget,
        temperature=temperature,
        top_k=top_k,
        stop_ids=stops,
        banned_ids=banned,
        seed=None if seed is None else seed + 1,
    ):
        answer_ids.append(token_id)
        budget.answer_used += 1
        text = tokenizer.decode(answer_ids, skip_special=True)
        delta = text[len(answer_text) :]
        answer_text = text
        if delta:
            yield {"type": "answer", "text": delta}
        if token_id == end_id:
            budget.answer_stop = "end"
            break

    if budget.answer_stop is None:
        budget.answer_stop = "budget"
    yield {"type": "done", "budget": budget.to_dict()}
