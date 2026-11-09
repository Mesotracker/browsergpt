from __future__ import annotations

import importlib
import json
import time
import zipfile

from fastapi.testclient import TestClient

from chainforge import cgpt
from tests.helpers import tiny_model, tiny_tokenizer


def _server(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAINFORGE_DATA", str(tmp_path / "data"))
    import chainforge.server as server

    server = importlib.reload(server)
    return server


def _tiny_archive(tmp_path) -> tuple[object, object]:
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, ctx=48)
    path = tmp_path / "tiny.cgpt"
    cgpt.save(path, model, tokenizer, name="tiny")
    return path, tokenizer


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def test_full_api_flow(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    archive, _tokenizer = _tiny_archive(tmp_path)

    with TestClient(server.app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/api/health").json()
        assert health["ok"] is True

        imported = client.post(
            "/api/models/import",
            files={"file": ("tiny.cgpt", archive.read_bytes(), "application/zip")},
        )
        assert imported.status_code == 200, imported.text
        assert imported.json()["model"]["name"].startswith("tiny")
        assert any(m["name"] == "tiny" for m in client.get("/api/models").json()["models"])

        download = client.get("/api/models/tiny/file")
        assert download.status_code == 200
        assert download.content[:2] == b"PK"

        dataset = client.post(
            "/api/datasets",
            files={
                "files": (
                    "corpus.txt",
                    ("ChainForge trains a mini GPT on documents. " * 400).encode(),
                    "text/plain",
                )
            },
        )
        assert dataset.status_code == 200, dataset.text
        dataset_id = dataset.json()["id"]
        assert dataset.json()["files"][0]["chars"] > 1000

        queued = client.post(
            "/api/train",
            json={
                "dataset_id": dataset_id,
                "mode": "major",
                "name": "api-train",
                "steps": 1,
                "batch_size": 1,
                "lr": 0.001,
                "seed": 3,
            },
        )
        assert queued.status_code == 200, queued.text
        job_id = queued.json()["job_id"]

        status = client.get(f"/api/train/{job_id}").json()
        deadline = time.time() + 45
        while status["status"] not in ("done", "error", "cancelled") and time.time() < deadline:
            time.sleep(0.2)
            status = client.get(f"/api/train/{job_id}").json()
        assert status["status"] == "done", status
        assert status["result"]["model"]["name"] == "api-train"

        chat = client.post(
            "/api/chat",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "What is ChainForge?"}],
                "reasoning": True,
                "think_tokens": 4,
                "answer_tokens": 4,
                "temperature": 0.0,
                "top_k": 1,
            },
        )
        assert chat.status_code == 200, chat.text
        events = _sse_events(chat.text)
        assert events[0]["type"] == "start"
        assert events[-1]["type"] == "done"
        budget = events[-1]["budget"]
        assert budget["context"] == 48
        assert budget["think_used"] <= budget["think_budget"]
        assert budget["answer_used"] <= budget["answer_budget"]


def test_import_rejects_hostile_archive(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    bad = tmp_path / "evil.cgpt"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("../../evil", "x")
    with TestClient(server.app) as client:
        response = client.post(
            "/api/models/import",
            files={"file": ("evil.cgpt", bad.read_bytes(), "application/zip")},
        )
        assert response.status_code == 422


def test_missing_model_returns_404(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    with TestClient(server.app) as client:
        assert client.get("/api/models/nope/file").status_code == 404
        response = client.post(
            "/api/chat",
            json={
                "model": "nope",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 404
