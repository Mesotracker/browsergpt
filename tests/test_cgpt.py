from __future__ import annotations

import hashlib
import json
import zipfile

import pytest
import torch

from chainforge import cgpt
from tests.helpers import tiny_model, tiny_tokenizer


def _roundtrip_model(tmp_path):
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, ctx=32)
    path = tmp_path / "tiny.cgpt"
    manifest = cgpt.save(
        path,
        model,
        tokenizer,
        name="tiny",
        training={"steps": 3, "final_loss": 1.5, "losses": [2.0, 1.7, 1.5]},
        lineage={"mode": "from_scratch", "parent": None},
    )
    return tokenizer, model, path, manifest


def test_export_import_reproduces_logits(tmp_path):
    tokenizer, model, path, manifest = _roundtrip_model(tmp_path)
    assert manifest["format"] == "cgpt/1.0"
    assert manifest["arch"]["vocab_size"] == tokenizer.vocab_size
    assert path.read_bytes()[:2] == b"PK"

    loaded = cgpt.load(path)
    assert loaded.tokenizer.vocab_size == tokenizer.vocab_size
    assert loaded.training["steps"] == 3
    assert loaded.manifest["lineage"]["mode"] == "from_scratch"

    token_ids = torch.randint(0, tokenizer.vocab_size, (2, 12))
    model.eval()
    loaded.model.eval()
    with torch.no_grad():
        expected, _ = model(token_ids)
        actual, _ = loaded.model(token_ids)
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)


def test_checksum_mismatch_is_rejected(tmp_path):
    _tokenizer, _model, path, manifest = _roundtrip_model(tmp_path)
    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}

    manifest["weights"]["sha256"] = "0" * 64
    members["manifest.json"] = json.dumps(manifest).encode()
    tampered = tmp_path / "tampered.cgpt"
    with zipfile.ZipFile(tampered, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)

    with pytest.raises(ValueError, match="checksum"):
        cgpt.load(tampered)


def test_extra_and_traversal_members_are_rejected(tmp_path):
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer)
    path = tmp_path / "bad.cgpt"
    cgpt.save(path, model, tokenizer, name="bad")
    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    members["../escape.txt"] = b"nope"
    bad = tmp_path / "escape.cgpt"
    with zipfile.ZipFile(bad, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    with pytest.raises(ValueError, match="unexpected archive member"):
        cgpt.load(bad)


def test_manifest_hash_is_stable(tmp_path):
    _tokenizer, _model, path, manifest = _roundtrip_model(tmp_path)
    assert cgpt.read_manifest(path)["id"] == manifest["id"]
    with zipfile.ZipFile(path) as zf:
        weights = zf.read("weights.safetensors")
    assert hashlib.sha256(weights).hexdigest() == manifest["weights"]["sha256"]
