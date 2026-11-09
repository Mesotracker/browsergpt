"""High-level training pipelines shared by the CLI, server and tests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import cgpt, ingest
from .config import PRESETS, ModelConfig, TrainConfig, make_model_config
from .model import GPT
from .paths import ROOT
from .tokenizer import ByteBPE
from .trainer import TokenData, TrainReport, pick_device, train_model


class PipelineError(ValueError):
    """A user-fixable error while preparing or running a training job."""


def collect_files(paths: Iterable[Path | str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.name.startswith("."):
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)
    return files


def prepare_dataset(
    documents: Sequence[ingest.Document],
    tokenizer: ByteBPE,
    out_dir: Path,
    *,
    val_frac: float = 0.02,
    rejected: Sequence[dict] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = ingest.write_token_bin(documents, tokenizer, out_dir / "tokens.bin")
    total = int(stats["total_tokens"])
    if total < 256:
        raise PipelineError(
            "the extracted documents produced too few tokens to train on; add more text"
        )
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tokens": total,
        "files": stats["files"],
        "rejected": list(rejected or []),
        "tokenizer_vocab": tokenizer.vocab_size,
        "val_frac": val_frac,
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def run_training(
    *,
    files: Sequence[Path | str],
    mode: str,
    output_path: Path | str,
    name: str,
    preset: str = "base",
    steps: int = 300,
    batch_size: int = 16,
    lr: float | None = None,
    ctx: int | None = None,
    vocab_size: int | None = None,
    seed: int = 1337,
    dropout: float | None = None,
    model_overrides: dict | None = None,
    init_model: Path | str | None = None,
    work_dir: Path | str | None = None,
    device: str = "auto",
    on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Train a new model from documents or fine-tune an existing ``.cgpt``."""
    if mode not in ("major", "finetune"):
        raise PipelineError("mode must be 'major' or 'finetune'")
    files = [Path(f) for f in files]
    if not files:
        raise PipelineError("no input files were supplied")

    work_dir = Path(work_dir or (ROOT / "data" / "runs" / f"run-{int(time.time())}"))
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    def emit(event: dict) -> None:
        if on_event is not None:
            on_event(event)

    documents, rejected = ingest.collect_documents(files)
    if not documents:
        detail = "; ".join(f"{r['name']}: {r['reason']}" for r in rejected[:5])
        raise PipelineError(f"no usable text was extracted. {detail}".strip())
    emit(
        {
            "type": "log",
            "message": f"Ingested {len(documents)} document(s) "
            f"({sum(len(d.text) for d in documents):,} chars)"
            + (f", rejected {len(rejected)}" if rejected else ""),
        }
    )

    text = "\n\n".join(d.text for d in documents)
    lineage: dict

    if mode == "finetune":
        if not init_model:
            raise PipelineError("fine-tuning requires a parent .cgpt model")
        loaded = cgpt.load(Path(init_model), device="cpu")
        model = loaded.model
        tokenizer = loaded.tokenizer
        cfg = loaded.cfg
        if ctx is not None and int(ctx) != cfg.ctx:
            raise PipelineError(
                f"fine-tuning keeps the parent context window ({cfg.ctx}); "
                "start a major run to change architecture"
            )
        lineage = {
            "mode": "finetune",
            "parent": loaded.manifest.get("name"),
            "parent_id": loaded.manifest.get("id"),
        }
        emit({"type": "log", "message": f"Fine-tuning from {loaded.manifest.get('name')}"})
    else:
        target_vocab = int(vocab_size or PRESETS.get(preset, PRESETS["base"])["vocab_size"])
        tokenizer = ByteBPE.train([text], vocab_size=target_vocab)
        overrides = dict(model_overrides or {})
        if ctx is not None:
            overrides["ctx"] = int(ctx)
        if dropout is not None:
            overrides["dropout"] = float(dropout)
        overrides["vocab_size"] = tokenizer.vocab_size
        try:
            cfg = make_model_config(preset, **overrides)
        except (ValueError, TypeError) as exc:
            raise PipelineError(f"invalid model configuration: {exc}") from exc
        model = GPT(cfg)
        lineage = {"mode": "from_scratch", "parent": None}
        emit(
            {
                "type": "log",
                "message": f"Tokenizer: {tokenizer.vocab_size:,} tokens; "
                f"model: {cfg.n_layer} layers / {cfg.d_model} dim / ctx {cfg.ctx}",
            }
        )

    dataset_manifest = prepare_dataset(
        documents,
        tokenizer,
        work_dir / "dataset",
        val_frac=0.02,
        rejected=rejected,
    )
    min_tokens = max(256, cfg.ctx * 3)
    if dataset_manifest["total_tokens"] < min_tokens:
        raise PipelineError(
            f"dataset has {dataset_manifest['total_tokens']:,} tokens; "
            f"at least {min_tokens:,} are needed for this context window"
        )

    train_cfg = TrainConfig(
        steps=max(1, int(steps)),
        batch_size=max(1, int(batch_size)),
        lr=float(lr if lr is not None else (1e-4 if mode == "finetune" else 4e-4)),
        seed=int(seed),
        eval_interval=max(1, min(50, max(1, int(steps) // 3))),
        eval_batches=4,
    )
    try:
        train_cfg.validate(cfg.ctx)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc

    resolved_device = pick_device(device)
    emit(
        {
            "type": "log",
            "message": f"Training {train_cfg.steps:,} steps on {resolved_device} "
            f"with {dataset_manifest['total_tokens']:,} tokens",
        }
    )
    data = TokenData(work_dir / "dataset" / "tokens.bin", val_frac=train_cfg.val_frac)
    report: TrainReport = train_model(
        model,
        data,
        train_cfg,
        device=resolved_device,
        on_event=on_event,
        should_stop=should_stop,
    )

    training_meta = {
        "mode": mode,
        "steps": train_cfg.steps,
        "batch_size": train_cfg.batch_size,
        "lr": train_cfg.lr,
        "ctx": cfg.ctx,
        "seed": train_cfg.seed,
        "device": resolved_device,
        "seconds": round(report.seconds, 2),
        "tokens_seen": report.tokens_seen,
        "final_loss": report.final_loss,
        "initial_loss": report.initial_loss,
        "best_val_loss": report.best_val_loss,
        "losses": [round(x, 5) for x in report.losses],
        "val_losses": report.val_losses,
        "dataset": {
            "total_tokens": dataset_manifest["total_tokens"],
            "documents": len(documents),
            "files": [
                {
                    "name": f["name"],
                    "chars": f["chars"],
                    "tokens": f["tokens"],
                }
                for f in dataset_manifest["files"]
            ],
            "rejected": rejected,
        },
    }
    manifest = cgpt.save(
        output_path,
        model,
        tokenizer,
        name=name,
        training=training_meta,
        lineage=lineage,
    )
    emit(
        {
            "type": "log",
            "message": f"Exported {name} ({manifest['params']:,} params, "
            f"{output_path.stat().st_size / 1e6:.1f} MB)",
        }
    )
    return {
        "path": str(output_path),
        "name": name,
        "manifest": manifest,
        "report": report.to_dict(),
        "dataset": dataset_manifest,
    }


def build_base(
    *,
    output_path: Path | str,
    steps: int = 300,
    preset: str = "base",
    corpus_dirs: Sequence[Path | str] | None = None,
    extra_files: Sequence[Path | str] = (),
    name: str = "base",
    device: str = "auto",
    on_event: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Build the starter base model from the bundled corpus."""
    corpus_dirs = list(corpus_dirs or [ROOT / "corpus" / "base"])
    paths: list[Path] = [Path(p) for p in corpus_dirs]
    paths.extend(Path(p) for p in extra_files)
    files = collect_files(paths)
    if not files:
        raise PipelineError("the corpus is empty; add documents to corpus/base")
    return run_training(
        files=files,
        mode="major",
        output_path=output_path,
        name=name,
        preset=preset,
        steps=steps,
        batch_size=16,
        lr=5e-4,
        device=device,
        on_event=on_event,
        should_stop=should_stop,
    )
