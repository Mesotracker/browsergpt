"""The ``.cgpt`` model container.

A ``.cgpt`` file is a zip archive holding plain JSON metadata, a JSON
tokenizer, and safetensors weights. No pickle is used anywhere, and imports are
validated against an allowlist with size, checksum and architecture bounds.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save

from .config import ModelConfig
from .model import GPT
from .tokenizer import ByteBPE

FORMAT = "cgpt/1.0"
MANIFEST = "manifest.json"
TOKENIZER = "tokenizer.json"
WEIGHTS = "weights.safetensors"
TRAINING = "training.json"
ALLOWED_MEMBERS = {MANIFEST, TOKENIZER, WEIGHTS, TRAINING}

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 400 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED = 1024 * 1024 * 1024
MAX_PARAMS = 60_000_000
MAX_MANIFEST_BYTES = 1024 * 1024


@dataclass
class LoadedModel:
    model: GPT
    tokenizer: ByteBPE
    cfg: ModelConfig
    manifest: dict
    training: dict
    path: Path | None = None

    @property
    def name(self) -> str:
        return str(self.manifest.get("name", "model"))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _compact_losses(losses: list[Any], limit: int = 600) -> list[Any]:
    if len(losses) <= limit:
        return losses
    stride = len(losses) / limit
    return [losses[min(len(losses) - 1, int(i * stride))] for i in range(limit)]


def save(
    path: Path,
    model: GPT,
    tokenizer: ByteBPE,
    *,
    name: str,
    training: dict | None = None,
    lineage: dict | None = None,
) -> dict:
    """Write a model archive atomically and return its manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg

    state = {
        key: value.detach().to("cpu").contiguous().float()
        for key, value in model.state_dict().items()
        if not (cfg.tie_embeddings and key == "head.weight")
    }
    weights_blob = safetensors_save(state)
    weights_sha = hashlib.sha256(weights_blob).hexdigest()

    training = dict(training or {})
    if isinstance(training.get("losses"), list):
        training["losses"] = _compact_losses(training["losses"])
    if isinstance(training.get("val_losses"), list):
        training["val_losses"] = _compact_losses(training["val_losses"], 300)

    manifest = {
        "format": FORMAT,
        "id": uuid.uuid4().hex,
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": cfg.to_dict(),
        "params": model.n_params,
        "tokenizer": {
            "type": "byte_bpe",
            "vocab_size": tokenizer.vocab_size,
            "specials": sorted(tokenizer.specials),
        },
        "weights": {
            "path": WEIGHTS,
            "sha256": weights_sha,
            "dtype": "float32",
        },
        "lineage": lineage or {"mode": "from_scratch", "parent": None},
        "caps": {
            "reasoning_protocol": True,
            "chat_protocol": "user/assistant/think/endthink/end",
        },
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST, _json_bytes(manifest))
        zf.writestr(TOKENIZER, _json_bytes(tokenizer.to_dict()))
        zf.writestr(WEIGHTS, weights_blob)
        zf.writestr(TRAINING, _json_bytes(training))
    os.replace(tmp_path, path)
    return manifest


def _validate_members(zf: zipfile.ZipFile) -> None:
    seen: set[str] = set()
    total = 0
    for info in zf.infolist():
        name = info.filename
        if name in seen:
            raise ValueError("duplicate archive member")
        seen.add(name)
        if name not in ALLOWED_MEMBERS:
            raise ValueError(f"unexpected archive member: {name!r}")
        if info.is_dir():
            raise ValueError("directories are not allowed in a .cgpt archive")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"{name} is too large")
        total += info.file_size
        if total > MAX_TOTAL_UNCOMPRESSED:
            raise ValueError("archive expands beyond the size limit")
        if info.compress_size and info.file_size > info.compress_size * 400:
            raise ValueError(f"{name} has an implausible compression ratio")
    missing = {MANIFEST, TOKENIZER, WEIGHTS} - seen
    if missing:
        raise ValueError(f"archive is missing: {', '.join(sorted(missing))}")


def _open_validated(path: Path) -> zipfile.ZipFile:
    path = Path(path)
    if not path.is_file():
        raise ValueError("model file does not exist")
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("model file is too large")
    zf = zipfile.ZipFile(path)
    try:
        _validate_members(zf)
    except Exception:
        zf.close()
        raise
    return zf


def read_manifest(path: Path) -> dict:
    """Read and validate the container metadata without loading weights."""
    with _open_validated(Path(path)) as zf:
        raw = zf.read(MANIFEST)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("manifest is too large")
        manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if str(manifest.get("format", "")).split("/")[0] != "cgpt":
        raise ValueError("not a .cgpt model")
    arch = manifest.get("arch")
    if not isinstance(arch, dict):
        raise ValueError("manifest is missing architecture metadata")
    cfg = ModelConfig.from_dict(arch)
    if cfg.estimate_params() > MAX_PARAMS:
        raise ValueError("model is larger than this build supports")
    return manifest


def load(path: Path, device: str = "cpu") -> LoadedModel:
    path = Path(path)
    with _open_validated(path) as zf:
        manifest = json.loads(zf.read(MANIFEST).decode("utf-8"))
        tokenizer_data = json.loads(zf.read(TOKENIZER).decode("utf-8"))
        training = {}
        try:
            training = json.loads(zf.read(TRAINING).decode("utf-8"))
        except KeyError:
            pass
        weights_blob = zf.read(WEIGHTS)

    if str(manifest.get("format", "")).split("/")[0] != "cgpt":
        raise ValueError("not a .cgpt model")
    cfg = ModelConfig.from_dict(manifest.get("arch") or {})
    if cfg.estimate_params() > MAX_PARAMS:
        raise ValueError("model is larger than this build supports")

    expected_sha = (manifest.get("weights") or {}).get("sha256")
    actual_sha = hashlib.sha256(weights_blob).hexdigest()
    if expected_sha and expected_sha != actual_sha:
        raise ValueError("weights checksum mismatch; the model file is corrupted")

    tokenizer = ByteBPE.from_dict(tokenizer_data)
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the model")

    state = safetensors_load(weights_blob)
    if not state:
        raise ValueError("model archive contains no weights")
    model = GPT(cfg)
    allowed_missing = {"head.weight"} if cfg.tie_embeddings else set()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = set(missing) - allowed_missing
    if missing or unexpected:
        raise ValueError(
            f"weights do not match the architecture (missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)})"
        )
    if cfg.tie_embeddings:
        model.head.weight = model.tok_emb.weight
    model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        manifest=manifest,
        training=training if isinstance(training, dict) else {},
        path=path,
    )


def inspect(path: Path) -> dict:
    path = Path(path)
    manifest = read_manifest(path)
    training = {}
    with _open_validated(path) as zf:
        try:
            training = json.loads(zf.read(TRAINING).decode("utf-8"))
        except KeyError:
            pass
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "manifest": manifest,
        "training": training,
    }
