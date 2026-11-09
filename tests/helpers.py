from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chainforge.config import ModelConfig
from chainforge.model import GPT
from chainforge.tokenizer import ByteBPE

CORPUS = (
    "ChainForge trains small language models from documents.\n"
    "The quick brown fox jumps over the lazy dog.\n"
    "Tokens become embeddings, attention mixes context, and the model predicts "
    "the next token.\n"
    "Fine-tuning continues from a saved model archive.\n"
    "A context window bounds every prompt and every generated token.\n"
) * 16


def tiny_tokenizer(vocab_size: int = 300) -> ByteBPE:
    return ByteBPE.train([CORPUS], vocab_size=vocab_size)


def tiny_model(tokenizer: ByteBPE, ctx: int = 32, seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            ctx=ctx,
            d_model=32,
            n_layer=1,
            n_head=2,
            dropout=0.0,
        )
    )


def write_token_file(path: Path, tokenizer: ByteBPE, repeats: int = 12) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = tokenizer.encode(CORPUS) * repeats
    array = np.asarray(ids, dtype=np.uint16)
    path.write_bytes(array.tobytes())
    return int(array.size)
