#!/usr/bin/env python3
"""Assemble the static ChainForge browser edition into _site/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"missing build input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    copy_file(ROOT / "pages" / "index.html", OUT / "index.html")
    copy_file(ROOT / "pages" / "extra.css", OUT / "extra.css")
    copy_file(ROOT / "web" / "app.css", OUT / "app.css")
    copy_file(ROOT / "web" / "favicon.svg", OUT / "favicon.svg")

    corpus_out = OUT / "corpus"
    corpus_out.mkdir()
    corpus_files = sorted((ROOT / "corpus" / "base").glob("*.md"))
    if not corpus_files:
        raise SystemExit("missing corpus/base/*.md seed files")
    for source in corpus_files:
        copy_file(source, corpus_out / source.name)

    base_model = ROOT / "pages" / "base-model.json"
    if base_model.is_file():
        copy_file(base_model, OUT / "base-model.json")
        print(f"base model: {base_model.stat().st_size / 1e6:.2f} MB")
    else:
        print("warning: pages/base-model.json is missing; the page will offer fresh training only", file=sys.stderr)

    # GitHub Pages otherwise runs Jekyll and can drop leading-underscore paths.
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"assembled {OUT} ({sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) / 1e6:.2f} MB before JS bundle)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
