"""Configuration objects shared by the trainer, server and CLI."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Architecture of a decoder-only transformer.

    Kept deliberately small: these models are trained on CPUs in CI, so a
    compact architecture matters more than raw parameter count.
    """

    vocab_size: int
    ctx: int = 256
    d_model: int = 160
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 256 <= self.vocab_size <= 65536:
            raise ValueError("vocab_size must be between 256 and 65536")
        if not 32 <= self.ctx <= 2048:
            raise ValueError("ctx must be between 32 and 2048")
        if not 16 <= self.d_model <= 2048:
            raise ValueError("d_model must be between 16 and 2048")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if not 1 <= self.n_head <= 64:
            raise ValueError("n_head must be between 1 and 64")
        if not 1 <= self.n_layer <= 48:
            raise ValueError("n_layer must be between 1 and 48")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def estimate_params(self) -> int:
        """Cheap parameter estimate used to reject hostile model archives."""
        d = self.d_model
        per_layer = 4 * d * d + 2 * d * 4 * d
        return self.vocab_size * d + self.ctx * d + self.n_layer * per_layer + 2 * d

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        allowed = {
            "vocab_size",
            "ctx",
            "d_model",
            "n_layer",
            "n_head",
            "dropout",
            "tie_embeddings",
        }
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 16
    lr: float = 4e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.1
    eval_interval: int = 50
    eval_batches: int = 4
    seed: int = 1337
    log_every: int = 10
    val_frac: float = 0.02

    def validate(self, ctx: int) -> None:
        if not 1 <= self.steps <= 200000:
            raise ValueError("steps must be between 1 and 200000")
        if not 1 <= self.batch_size <= 512:
            raise ValueError("batch_size must be between 1 and 512")
        if not 1e-6 <= self.lr <= 1.0:
            raise ValueError("lr must be between 1e-6 and 1.0")
        if self.eval_interval < 1:
            raise ValueError("eval_interval must be >= 1")
        if self.eval_batches < 1:
            raise ValueError("eval_batches must be >= 1")
        if not 0.0 <= self.val_frac < 0.5:
            raise ValueError("val_frac must be in [0, 0.5)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        allowed = {
            "steps",
            "batch_size",
            "lr",
            "weight_decay",
            "grad_clip",
            "warmup_frac",
            "min_lr_frac",
            "eval_interval",
            "eval_batches",
            "seed",
            "log_every",
            "val_frac",
        }
        return cls(**{k: v for k, v in data.items() if k in allowed})


# Architecture presets. `base` trains comfortably on a GitHub Actions CPU
# runner; `wide` is the next step up if you have a GPU or a longer job.
PRESETS: dict[str, dict[str, Any]] = {
    "base": {
        "ctx": 256,
        "d_model": 160,
        "n_layer": 4,
        "n_head": 4,
        "dropout": 0.1,
        "vocab_size": 1024,
    },
    "wide": {
        "ctx": 384,
        "d_model": 384,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.1,
        "vocab_size": 4096,
    },
}


def make_model_config(preset: str = "base", **overrides: Any) -> ModelConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = dict(PRESETS[preset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return ModelConfig(**cfg)


def cosine_lr(step: int, total: int, base_lr: float, warmup_frac: float, min_frac: float) -> float:
    """Linear warmup followed by cosine decay to ``base_lr * min_frac``."""
    total = max(total, 1)
    warmup = max(1, int(total * warmup_frac))
    if step <= warmup:
        return base_lr * (0.05 + 0.95 * step / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    min_lr = base_lr * min_frac
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, progress)))
