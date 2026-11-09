"""Command line interface for ChainForge."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from . import __version__, cgpt, pipeline
from .config import PRESETS
from .generate import chat_stream


def _event_printer(json_mode: bool):
    def on_event(event: dict) -> None:
        if event.get("type") == "step":
            print(
                f"step {event['step']:>5}/{event['total']}  "
                f"loss {event['loss']:.4f}  lr {event['lr']:.2e}  "
                f"{event['tok_s']:,} tok/s  eta {event['eta_s']}s",
                file=sys.stderr,
                flush=True,
            )
        elif event.get("type") == "eval":
            print(
                f"  eval @ {event['step']}: val loss {event['val_loss']:.4f}",
                file=sys.stderr,
                flush=True,
            )
        elif event.get("type") == "log" and not json_mode:
            print(f"  · {event['message']}", file=sys.stderr, flush=True)

    return on_event


def _cmd_train(args: argparse.Namespace) -> int:
    files = pipeline.collect_files(args.files)
    if not files:
        print("error: no input files found", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.suffix != ".cgpt":
        out = out.with_suffix(".cgpt")
    try:
        result = pipeline.run_training(
            files=files,
            mode=args.mode,
            output_path=out,
            name=args.name or out.stem,
            preset=args.preset,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            ctx=args.ctx,
            vocab_size=args.vocab_size,
            dropout=args.dropout,
            seed=args.seed,
            init_model=args.init,
            work_dir=args.work_dir,
            device=args.device,
            on_event=_event_printer(args.json),
        )
    except (pipeline.PipelineError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = {
        "model": result["name"],
        "path": str(out),
        "params": result["manifest"]["params"],
        "steps": result["report"]["steps"],
        "final_loss": result["report"]["final_loss"],
        "val_loss": result["report"]["best_val_loss"],
        "tokens": result["report"]["tokens_seen"],
        "seconds": round(result["report"]["seconds"], 2),
        "dataset_tokens": result["dataset"]["total_tokens"],
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"\nSaved {out} — {summary['params']:,} params, "
            f"final loss {summary['final_loss']:.4f}, "
            f"{summary['seconds']:.1f}s"
        )
    return 0


def _cmd_build_base(args: argparse.Namespace) -> int:
    out = Path(args.out)
    corpus_dirs = args.corpus or [pipeline.ROOT / "corpus" / "base"]
    extra: list[Path] = [Path(p) for p in (args.extra or [])]
    if args.include_docs:
        extra.extend(
            [pipeline.ROOT / "README.md", *sorted((pipeline.ROOT / "docs").glob("*.md"))]
        )
    try:
        result = pipeline.build_base(
            output_path=out,
            steps=args.steps,
            preset=args.preset,
            corpus_dirs=corpus_dirs,
            extra_files=[path for path in extra if path.is_file()],
            device=args.device,
            on_event=_event_printer(args.json),
        )
    except (pipeline.PipelineError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = {
        "model": result["name"],
        "path": str(out),
        "params": result["manifest"]["params"],
        "steps": result["report"]["steps"],
        "final_loss": result["report"]["final_loss"],
        "val_loss": result["report"]["best_val_loss"],
        "seconds": round(result["report"]["seconds"], 2),
        "dataset_tokens": result["dataset"]["total_tokens"],
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Base model written to {out} — {summary['params']:,} params, "
            f"final loss {summary['final_loss']:.4f}"
        )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        data = cgpt.inspect(Path(args.model))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data, indent=2))
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    try:
        loaded = cgpt.load(Path(args.model))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    messages = [{"role": "user", "content": args.prompt}]
    reasoning = not args.no_reasoning
    trace: list[str] = []
    answer: list[str] = []
    budget: dict = {}
    for event in chat_stream(
        loaded.model,
        loaded.tokenizer,
        messages,
        reasoning=reasoning,
        think_tokens=args.think,
        answer_tokens=args.answer,
        temperature=args.temperature,
        top_k=args.top_k,
    ):
        if event["type"] == "start":
            budget = event.get("budget", {})
        elif event["type"] == "reasoning":
            trace.append(event["text"])
        elif event["type"] == "answer":
            answer.append(event["text"])
        elif event["type"] == "done":
            budget = event.get("budget", budget)
    if trace:
        print("--- reasoning draft ---")
        print("".join(trace).strip())
        print("--- answer ---")
    print("".join(answer).strip() or "(the model produced no answer tokens)")
    if args.json:
        print(
            json.dumps(
                {
                    "reasoning": "".join(trace),
                    "answer": "".join(answer),
                    "budget": budget,
                }
            )
        )
    else:
        print(
            f"\n[prompt {budget.get('prompt_tokens')} · "
            f"reasoning {budget.get('think_used')}/{budget.get('think_budget')} · "
            f"answer {budget.get('answer_used')}/{budget.get('answer_budget')} · "
            f"ctx {budget.get('context')}]",
            file=sys.stderr,
        )
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="chainforge-smoke-") as tmp:
        tmp_path = Path(tmp)
        seed_text = (
            "ChainForge trains small transformer models on your own documents. "
            "Documents can be text, markdown, Word files or text-layer PDFs. "
            "A training run packs tokens into a flat stream and learns to predict "
            "the next token. Fine-tuning starts from an existing model archive. "
            "The chat protocol includes a visible reasoning draft and a token budget. "
        ) * 40
        corpus = tmp_path / "corpus.md"
        corpus.write_text(seed_text, encoding="utf-8")
        out = tmp_path / "smoke.cgpt"
        print("smoke: training a tiny model …", file=sys.stderr)
        try:
            result = pipeline.run_training(
                files=[corpus],
                mode="major",
                output_path=out,
                name="smoke",
                steps=args.steps,
                batch_size=4,
                lr=1e-3,
                work_dir=tmp_path / "run",
                model_overrides={
                    "d_model": 48,
                    "n_layer": 1,
                    "n_head": 2,
                    "ctx": 48,
                    "vocab_size": 320,
                },
                on_event=_event_printer(False),
            )
            loaded = cgpt.load(out)
            events = list(
                chat_stream(
                    loaded.model,
                    loaded.tokenizer,
                    [{"role": "user", "content": "What is ChainForge?"}],
                    reasoning=True,
                    think_tokens=8,
                    answer_tokens=16,
                    temperature=0.8,
                )
            )
        except Exception as exc:  # noqa: BLE001 - smoke test prints the failure
            print(f"smoke failed: {exc}", file=sys.stderr)
            return 1
        if not any(event["type"] == "done" for event in events):
            print("smoke failed: chat stream did not finish", file=sys.stderr)
            return 1
        summary = {
            "ok": True,
            "params": result["manifest"]["params"],
            "final_loss": result["report"]["final_loss"],
            "model_bytes": out.stat().st_size,
            "chat_events": len(events),
            "reasoning_draft": "".join(
                e["text"] for e in events if e["type"] == "reasoning"
            )[:200],
        }
        print(json.dumps(summary, indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "chainforge.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chainforge",
        description="Train and chat with a custom mini-GPT.",
    )
    parser.add_argument("--version", action="version", version=f"chainforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a model from documents")
    train.add_argument("files", nargs="+", help="files or directories to train on")
    train.add_argument("--out", required=True, help="output .cgpt path")
    train.add_argument("--name", help="model name stored inside the archive")
    train.add_argument("--mode", choices=["major", "finetune"], default="major")
    train.add_argument("--init", help="parent .cgpt model for --mode finetune")
    train.add_argument("--preset", choices=sorted(PRESETS), default="base")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float)
    train.add_argument("--ctx", type=int)
    train.add_argument("--vocab-size", type=int)
    train.add_argument("--dropout", type=float)
    train.add_argument("--seed", type=int, default=1337)
    train.add_argument("--device", default="auto")
    train.add_argument("--work-dir", help="directory for tokenized data")
    train.add_argument("--json", action="store_true", help="emit a JSON summary")
    train.set_defaults(func=_cmd_train)

    build = sub.add_parser("build-base", help="build the starter base model")
    build.add_argument("--out", default="models/base.cgpt")
    build.add_argument("--steps", type=int, default=300)
    build.add_argument("--preset", choices=sorted(PRESETS), default="base")
    build.add_argument("--corpus", nargs="*", help="corpus directories")
    build.add_argument("--extra", nargs="*", help="extra files to include")
    build.add_argument("--include-docs", action="store_true", default=True)
    build.add_argument("--device", default="auto")
    build.add_argument("--json", action="store_true")
    build.set_defaults(func=_cmd_build_base)

    inspect = sub.add_parser("inspect", help="print .cgpt metadata")
    inspect.add_argument("model")
    inspect.set_defaults(func=_cmd_inspect)

    chat = sub.add_parser("chat", help="chat with a .cgpt model in the terminal")
    chat.add_argument("--model", required=True)
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--think", type=int, default=48)
    chat.add_argument("--answer", type=int, default=96)
    chat.add_argument("--temperature", type=float, default=0.85)
    chat.add_argument("--top-k", type=int, default=40)
    chat.add_argument("--no-reasoning", action="store_true")
    chat.add_argument("--json", action="store_true")
    chat.set_defaults(func=_cmd_chat)

    smoke = sub.add_parser("smoke", help="verify train, export and chat end to end")
    smoke.add_argument("--steps", type=int, default=8)
    smoke.set_defaults(func=_cmd_smoke)

    serve = sub.add_parser("serve", help="run the web app")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.time()
    code = int(args.func(args))
    if (
        code == 0
        and args.command in ("train", "build-base")
        and not getattr(args, "json", False)
    ):
        print(f"(finished in {time.time() - started:.1f}s)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
