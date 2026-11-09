"""Training loop and token batching."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import TrainConfig, cosine_lr
from .model import GPT


class TrainingCancelled(Exception):
    """Raised when a job cancellation is requested."""


@dataclass
class TokenData:
    """Memory-mapped uint16 token stream with a train/validation split."""

    path: Path
    val_frac: float = 0.02
    _array: np.memmap | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.is_file():
            raise ValueError(f"token file not found: {self.path}")
        total = self.path.stat().st_size // 2
        if total < 64:
            raise ValueError("dataset is too small to train on")
        self.total_tokens = total
        self.val_len = min(total - 1, max(1, int(total * self.val_frac)))
        self.train_len = total - self.val_len

    @property
    def array(self) -> np.memmap:
        if self._array is None:
            self._array = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._array

    def batch(
        self,
        batch_size: int,
        ctx: int,
        split: str,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        length = self.train_len if split == "train" else self.val_len
        offset = 0 if split == "train" else self.train_len
        ctx = min(ctx, max(2, length - 1))
        max_start = max(0, length - ctx)
        starts = (
            rng.integers(0, max_start, size=batch_size)
            if max_start > 0
            else np.zeros(batch_size, dtype=np.int64)
        )
        rows = [
            np.asarray(self.array[offset + int(start) : offset + int(start) + ctx + 1])
            for start in starts
        ]
        stacked = np.stack(rows).astype(np.int64)
        x = torch.from_numpy(stacked[:, :-1])
        y = torch.from_numpy(stacked[:, 1:])
        return x, y


@dataclass
class TrainReport:
    steps: int
    losses: list[float]
    val_losses: list[dict]
    final_loss: float
    initial_loss: float
    best_val_loss: float | None
    tokens_seen: int
    seconds: float
    device: str
    params: int

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "losses": self.losses,
            "val_losses": self.val_losses,
            "final_loss": self.final_loss,
            "initial_loss": self.initial_loss,
            "best_val_loss": self.best_val_loss,
            "tokens_seen": self.tokens_seen,
            "seconds": self.seconds,
            "device": self.device,
            "params": self.params,
        }


def pick_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def evaluate(
    model: GPT,
    data: TokenData,
    cfg: TrainConfig,
    device: str,
) -> float:
    model.eval()
    rng = np.random.default_rng(cfg.seed + 999)
    total = 0.0
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            xb, yb = data.batch(cfg.batch_size, model.cfg.ctx, "val", rng)
            xb, yb = xb.to(device), yb.to(device)
            _logits, loss = model(xb, yb)
            total += float(loss.item())
    return total / max(1, cfg.eval_batches)


def train_model(
    model: GPT,
    data: TokenData,
    cfg: TrainConfig,
    *,
    device: str = "cpu",
    on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainReport:
    cfg.validate(model.cfg.ctx)
    torch.manual_seed(cfg.seed)
    model.to(device)
    model.train()
    optimizer = model.configure_optimizers(cfg.lr, cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    tokens_per_step = cfg.batch_size * model.cfg.ctx

    losses: list[float] = []
    val_losses: list[dict] = []
    best_val: float | None = None
    started = time.time()

    for step in range(1, cfg.steps + 1):
        if should_stop is not None and should_stop():
            raise TrainingCancelled("training cancelled")
        lr = cosine_lr(step, cfg.steps, cfg.lr, cfg.warmup_frac, cfg.min_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb = data.batch(cfg.batch_size, model.cfg.ctx, "train", rng)
        xb, yb = xb.to(device), yb.to(device)
        _logits, loss = model(xb, yb)
        if loss is None or not math.isfinite(float(loss.item())):
            raise RuntimeError("training diverged (non-finite loss)")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        loss_value = float(loss.item())
        losses.append(loss_value)

        if on_event is not None and (
            step == 1 or step == cfg.steps or step % cfg.log_every == 0
        ):
            elapsed = max(1e-6, time.time() - started)
            on_event(
                {
                    "type": "step",
                    "step": step,
                    "total": cfg.steps,
                    "loss": round(loss_value, 5),
                    "lr": lr,
                    "tokens": step * tokens_per_step,
                    "tok_s": int(step * tokens_per_step / elapsed),
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": int((elapsed / step) * (cfg.steps - step)),
                }
            )

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            model.eval()
            val_loss = evaluate(model, data, cfg, device)
            model.train()
            val_losses.append({"step": step, "val_loss": round(val_loss, 5)})
            best_val = val_loss if best_val is None else min(best_val, val_loss)
            if on_event is not None:
                on_event(
                    {
                        "type": "eval",
                        "step": step,
                        "val_loss": round(val_loss, 5),
                        "best_val_loss": round(best_val, 5),
                    }
                )

    model.eval()
    seconds = time.time() - started
    final_loss = losses[-1] if losses else float("nan")
    report = TrainReport(
        steps=cfg.steps,
        losses=losses,
        val_losses=val_losses,
        final_loss=final_loss,
        initial_loss=losses[0] if losses else final_loss,
        best_val_loss=best_val,
        tokens_seen=cfg.steps * tokens_per_step,
        seconds=seconds,
        device=device,
        params=model.n_params,
    )
    return report
