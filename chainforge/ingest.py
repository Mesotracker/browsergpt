"""Document ingestion: turn mixed uploads into clean training text.

Supports plain text and source files, Markdown, CSV/JSON/YAML, HTML/XML,
Word ``.docx`` and text-layer PDFs. Unknown files are decoded with a printable
ratio check instead of being silently accepted; binaries get a clear error.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .tokenizer import ByteBPE

MAX_CHARS_PER_FILE = 8_000_000

TEXT_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".conf", ".env.example", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx",
    ".jsx", ".html", ".htm", ".xml", ".css", ".scss", ".sass", ".sql",
    ".sh", ".bash", ".zsh", ".c", ".h", ".hpp", ".cc", ".cpp", ".java",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".kts", ".tex", ".bib",
    ".rtf", ".srt", ".vtt", ".diff", ".patch", ".gitignore", ".dockerfile",
}

REJECT_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg",
    ".mp3", ".wav", ".ogg", ".flac", ".m4a", ".mp4", ".mov", ".avi",
    ".mkv", ".webm", ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".class", ".jar", ".pt",
    ".pth", ".ckpt", ".safetensors", ".onnx", ".pickle", ".pkl", ".npy",
    ".npz", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".ico", ".psd",
    ".doc", ".xls", ".ppt", ".pptx", ".xlsx", ".epub", ".dmg", ".iso",
}

REJECT_REASONS = {
    ".pdf": "PDF extraction failed",
    ".doc": "legacy .doc is not supported; save as .docx or .txt",
    ".xls": "spreadsheets are not supported; export to CSV",
    ".xlsx": "spreadsheets are not supported; export to CSV",
    ".pptx": "presentations are not supported; export slides to text",
}


class IngestError(ValueError):
    """A file could not be turned into usable text."""


@dataclass
class Document:
    name: str
    text: str
    size: int = 0
    warnings: list[str] = field(default_factory=list)
    sha256: str = ""


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: D401 - element hook
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        elif tag in ("p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _printable_ratio(text: str) -> float:
    sample = text[:200_000]
    if not sample:
        return 0.0
    allowed = sum(1 for ch in sample if ch.isprintable() or ch in "\n\t")
    return allowed / len(sample)


def _decode_generic(data: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        try:
            return data.decode("utf-16"), warnings
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = data.decode(encoding)
            if _printable_ratio(text) < 0.8:
                break
            return text, warnings
        except UnicodeDecodeError:
            continue
    try:
        text = data.decode("cp1252")
        if _printable_ratio(text) >= 0.8:
            warnings.append("decoded as cp1252; original encoding was not UTF-8")
            return text, warnings
    except UnicodeDecodeError:
        pass
    raise IngestError(
        "file looks binary (no readable text). Images, audio and archives need OCR/transcription first"
    )


def _extract_docx(data: bytes) -> tuple[str, list[str]]:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise IngestError("python-docx is not installed") from exc
    try:
        doc = DocxDocument(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(f"could not read .docx: {exc}") from exc
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    warnings = []
    if not any(p.strip() for p in parts):
        warnings.append("the .docx contained no paragraph text")
    return "\n".join(parts), warnings


def _extract_pdf(data: bytes) -> tuple[str, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise IngestError("pypdf is not installed") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise IngestError(f"could not read .pdf: {exc}") from exc
    warnings: list[str] = []
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            warnings.append("a PDF page could not be parsed")
    text = "\n".join(parts)
    if not text.strip():
        raise IngestError(
            "PDF has no text layer (scanned PDFs would need OCR first)"
        )
    return text, warnings


def _extract_html(data: bytes) -> tuple[str, list[str]]:
    text, warnings = _decode_generic(data)
    parser = _HTMLText()
    try:
        parser.feed(text)
    except Exception as exc:
        raise IngestError(f"could not parse HTML: {exc}") from exc
    return parser.text(), warnings


def _extract_ipynb(data: bytes) -> tuple[str, list[str]]:
    try:
        notebook = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestError(f"could not parse notebook: {exc}") from exc
    cells = notebook.get("cells", [])
    parts: list[str] = []
    for cell in cells:
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str) or not source.strip():
            continue
        kind = cell.get("cell_type", "code")
        parts.append(f"### {kind}\n{source}")
    return "\n\n".join(parts), []


def _extract_rtf(text: str) -> tuple[str, list[str]]:
    cleaned = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
    cleaned = cleaned.replace("{", "").replace("}", "")
    return cleaned, ["RTF formatting was stripped"]


def extract_bytes(data: bytes, filename: str) -> Document:
    """Extract readable text from an uploaded file."""
    if not data:
        raise IngestError("file is empty")
    name = Path(filename or "upload.txt").name
    ext = Path(name).suffix.lower()
    if ext in REJECT_EXTENSIONS:
        raise IngestError(
            REJECT_REASONS.get(ext, f"{ext} files are not text; convert them first")
        )

    warnings: list[str] = []
    if ext == ".docx":
        text, warnings = _extract_docx(data)
    elif ext == ".pdf":
        text, warnings = _extract_pdf(data)
    elif ext in (".html", ".htm", ".xml"):
        text, warnings = _extract_html(data)
    elif ext == ".ipynb":
        text, warnings = _extract_ipynb(data)
    elif ext in TEXT_EXTENSIONS or ext == "":
        text, warnings = _decode_generic(data)
        if ext == ".rtf":
            text, extra = _extract_rtf(text)
            warnings.extend(extra)
    else:
        text, warnings = _decode_generic(data)
        warnings.append(f"unknown extension {ext or '(none)'}; decoded as plain text")

    text = normalize_text(text)
    if not text:
        raise IngestError("no text could be extracted")
    if len(text) > MAX_CHARS_PER_FILE:
        text = text[:MAX_CHARS_PER_FILE]
        warnings.append(f"truncated to {MAX_CHARS_PER_FILE:,} characters")
    return Document(
        name=name,
        text=text,
        size=len(data),
        warnings=warnings,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def extract_path(path: Path) -> Document:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IngestError(f"could not read {path.name}: {exc}") from exc
    return extract_bytes(data, path.name)


def dedupe_documents(documents: Iterable[Document]) -> tuple[list[Document], list[str]]:
    seen: dict[str, str] = {}
    unique: list[Document] = []
    skipped: list[str] = []
    for doc in documents:
        digest = doc.sha256 or hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
        if digest in seen:
            skipped.append(f"{doc.name} (duplicate of {seen[digest]})")
            continue
        seen[digest] = doc.name
        unique.append(doc)
    return unique, skipped


def collect_documents(paths: Sequence[Path]) -> tuple[list[Document], list[dict]]:
    documents: list[Document] = []
    rejected: list[dict] = []
    for path in paths:
        try:
            documents.append(extract_path(path))
        except IngestError as exc:
            rejected.append({"name": path.name, "reason": str(exc)})
    unique, skipped = dedupe_documents(documents)
    for item in skipped:
        rejected.append({"name": item, "reason": "duplicate content"})
    return unique, rejected


def write_token_bin(
    documents: Sequence[Document],
    tokenizer: ByteBPE,
    out_path: Path,
    end_token: str = "<|end|>",
) -> dict:
    """Tokenize documents into a flat uint16 binary stream.

    Writing per document keeps memory bounded even for a large dataset.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    end_id = tokenizer.specials.get(end_token)
    if end_id is None:
        raise ValueError(f"tokenizer is missing {end_token!r}")
    total = 0
    per_file: list[dict] = []
    with open(out_path, "wb") as handle:
        for doc in documents:
            ids = tokenizer.encode(doc.text)
            if not ids:
                continue
            row = np.asarray(ids + [end_id], dtype=np.uint16)
            handle.write(row.tobytes())
            total += len(row)
            per_file.append(
                {
                    "name": doc.name,
                    "bytes": doc.size,
                    "chars": len(doc.text),
                    "tokens": len(row),
                    "sha256": doc.sha256,
                    "warnings": doc.warnings,
                }
            )
    return {"total_tokens": total, "files": per_file}
