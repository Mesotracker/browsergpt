from __future__ import annotations

from chainforge.tokenizer import ByteBPE


def test_training_respects_vocab_budget_and_is_deterministic():
    text = ("ChainForge teaches a tiny model to write. " * 40) + ("héllo 世界 🚀\n" * 20)
    first = ByteBPE.train([text], vocab_size=300)
    second = ByteBPE.train([text], vocab_size=300)
    assert first.vocab_size <= 300
    assert first.vocab_size > 256
    assert first.merges == second.merges


def test_unicode_and_special_token_roundtrip():
    tokenizer = ByteBPE.train([("héllo 世界 🚀 ChainForge " * 30)], vocab_size=320)
    text = "héllo 世界 🚀 <|user|>question<|end|>"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.special_id("<|user|>") in ids
    assert tokenizer.decode(ids, skip_special=True) == "héllo 世界 🚀 question"


def test_unknown_bytes_are_still_encodable():
    tokenizer = ByteBPE.train(["plain english text " * 40], vocab_size=280)
    text = "Ω≈ç√∫˜µ≤≥÷ — unicode the tokenizer never saw"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_serialization_roundtrip():
    tokenizer = ByteBPE.train(["serialization matters " * 40], vocab_size=300)
    clone = ByteBPE.from_dict(tokenizer.to_dict())
    text = "restored <|bos|>weights<|end|>"
    assert clone.encode(text) == tokenizer.encode(text)
    assert clone.decode(clone.encode(text)) == text
