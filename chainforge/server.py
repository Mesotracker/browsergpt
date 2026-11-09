"""FastAPI application: training jobs, chat streaming, and the web UI."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import torch
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, cgpt, ingest, pipeline
from .config import PRESETS
from .generate import chat_stream
from .jobs import Job, JobManager, sse
from .paths import ROOT
from .store import ModelStore, safe_name

WEB_DIR = Path(os.environ.get("CHAINFORGE_WEB", ROOT / "web"))
DATA_DIR = Path(os.environ.get("CHAINFORGE_DATA", ROOT / "data"))
MODEL_DIR = DATA_DIR / "models"
DATASET_DIR = DATA_DIR / "datasets"
RUN_DIR = DATA_DIR / "runs"
TMP_DIR = DATA_DIR / "tmp"
for _directory in (WEB_DIR, MODEL_DIR, DATASET_DIR, RUN_DIR, TMP_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

ACCESS_TOKEN = os.environ.get("CHAINFORGE_TOKEN", "").strip()
MAX_FILE_MB = int(os.environ.get("CHAINFORGE_MAX_UPLOAD_MB", "25"))
MAX_FILES = int(os.environ.get("CHAINFORGE_MAX_FILES", "24"))
MAX_TOTAL_MB = int(os.environ.get("CHAINFORGE_MAX_TOTAL_UPLOAD_MB", "150"))
MAX_MODEL_MB = int(os.environ.get("CHAINFORGE_MAX_MODEL_MB", "256"))
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_TOTAL_BYTES = MAX_TOTAL_MB * 1024 * 1024

STORE = ModelStore(MODEL_DIR, ROOT / "models")
JOBS = JobManager()
_MODEL_CACHE: dict[str, tuple[float, cgpt.LoadedModel]] = {}
_CACHE_LOCK = threading.Lock()
_RATE_LOCK = threading.Lock()
_RATE: dict[str, list[float]] = defaultdict(list)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    JOBS.start()
    yield


app = FastAPI(title="ChainForge", version=__version__, lifespan=lifespan)


def require_auth(
    x_chainforge_token: str | None = Header(default=None, alias="X-Chainforge-Token"),
) -> None:
    if ACCESS_TOKEN and x_chainforge_token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing access token")


def _rate_limit(request: Request, scope: str, limit: int, window: float) -> None:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else ""
    if not ip and request.client is not None:
        ip = request.client.host
    key = f"{scope}:{ip or 'local'}"
    now = time.time()
    with _RATE_LOCK:
        bucket = [stamp for stamp in _RATE.get(key, []) if now - stamp < window]
        if len(bucket) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit reached ({limit} per {int(window)}s)",
            )
        bucket.append(now)
        _RATE[key] = bucket


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "upload.txt")
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return (base or "upload.txt")[:120]


def _load_model(name: str) -> cgpt.LoadedModel:
    try:
        path = STORE.resolve(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"model {name!r} not found") from exc
    key = path.name
    mtime = path.stat().st_mtime
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        loaded = cgpt.load(path, device=pipeline.pick_device())
    except Exception as exc:  # noqa: BLE001 - surface validation errors to the UI
        raise HTTPException(status_code=422, detail=f"could not load model: {exc}") from exc
    with _CACHE_LOCK:
        if len(_MODEL_CACHE) >= 2:
            _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
        _MODEL_CACHE[key] = (mtime, loaded)
    return loaded


def _compact_report(report: dict) -> dict:
    keys = (
        "steps",
        "final_loss",
        "initial_loss",
        "best_val_loss",
        "tokens_seen",
        "seconds",
        "device",
        "params",
    )
    summary = {key: report.get(key) for key in keys}
    summary["losses"] = report.get("losses", [])[-300:]
    summary["val_losses"] = report.get("val_losses", [])
    return summary


def _run_train_job(job: Job) -> dict:
    request = job.request
    dataset_dir = DATASET_DIR / request["dataset_id"] / "raw"
    if not dataset_dir.is_dir():
        raise ValueError("dataset was not found; upload the files again")
    files = sorted(path for path in dataset_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError("dataset contains no files")

    init_path = None
    if request["mode"] == "finetune":
        try:
            init_path = STORE.resolve(request["parent"])
        except FileNotFoundError as exc:
            raise ValueError(f"parent model {request['parent']!r} not found") from exc

    run_dir = RUN_DIR / job.id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "model.cgpt"
    name = request.get("name") or f"{request['mode']}-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        result = pipeline.run_training(
            files=files,
            mode=request["mode"],
            output_path=output_path,
            name=name,
            preset=request["preset"],
            steps=request["steps"],
            batch_size=request["batch_size"],
            lr=request.get("lr"),
            ctx=request.get("ctx"),
            vocab_size=request.get("vocab_size"),
            seed=request["seed"],
            init_model=init_path,
            work_dir=run_dir,
            device=pipeline.pick_device(),
            on_event=job.emit,
            should_stop=job.cancel_event.is_set,
        )
        stored = STORE.add_file(output_path, name)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    dataset = result.get("dataset", {})
    return {
        "model": {
            "name": stored["name"],
            "id": stored.get("id"),
            "params": stored.get("params"),
            "lineage": stored.get("lineage"),
        },
        "report": _compact_report(result["report"]),
        "dataset": {
            "total_tokens": dataset.get("total_tokens"),
            "documents": len(dataset.get("files", [])),
            "rejected": dataset.get("rejected", []),
        },
    }


def _run_base_job(job: Job) -> dict:
    run_dir = RUN_DIR / job.id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "base.cgpt"
    extra = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md"))
    try:
        result = pipeline.build_base(
            output_path=output_path,
            steps=int(job.request.get("steps", 300)),
            preset=str(job.request.get("preset", "base")),
            extra_files=[path for path in extra if path.is_file()],
            device=pipeline.pick_device(),
            on_event=job.emit,
            should_stop=job.cancel_event.is_set,
        )
        stored = STORE.add_file(output_path, "base")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return {
        "model": {
            "name": stored["name"],
            "id": stored.get("id"),
            "params": stored.get("params"),
            "lineage": stored.get("lineage"),
        },
        "report": _compact_report(result["report"]),
        "dataset": result.get("dataset", {}),
    }


# --------------------------------------------------------------------- models
class ImportResponse(BaseModel):
    model: dict


class TrainRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)
    mode: Literal["major", "finetune"] = "major"
    name: str | None = Field(default=None, max_length=64)
    preset: Literal["base", "wide"] = "base"
    steps: int = Field(default=300, ge=1, le=20000)
    batch_size: int = Field(default=16, ge=1, le=64)
    lr: float | None = Field(default=None, ge=1e-6, le=1e-2)
    ctx: int | None = Field(default=None, ge=32, le=512)
    vocab_size: int | None = Field(default=None, ge=260, le=8192)
    parent: str | None = Field(default=None, max_length=64)
    seed: int = Field(default=1337, ge=0, le=2**31 - 1)


class BaseRequest(BaseModel):
    steps: int = Field(default=300, ge=20, le=5000)
    preset: Literal["base", "wide"] = "base"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    model: str = Field(min_length=1, max_length=64)
    messages: list[ChatMessage] = Field(min_length=1, max_length=24)
    reasoning: bool = True
    think_tokens: int = Field(default=48, ge=0, le=512)
    answer_tokens: int = Field(default=96, ge=1, le=1024)
    temperature: float = Field(default=0.85, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=1000)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


router = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])


@router.get("/models")
def list_models() -> dict:
    models = STORE.list_models()
    for model in models:
        model["bundled"] = bool(model.get("bundled"))
    return {"models": models}


@router.post("/models/import", response_model=ImportResponse)
async def import_model(request: Request, file: UploadFile = File(...)) -> dict:
    _rate_limit(request, "import", 10, 3600)
    data = await file.read(MAX_MODEL_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MODEL_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="model file is too large")
    tmp = TMP_DIR / f"{uuid.uuid4().hex}.cgpt"
    try:
        tmp.write_bytes(data)
        cgpt.load(tmp)
        stem = Path(file.filename or "imported").stem
        stored = STORE.add_file(tmp, stem)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"invalid .cgpt file: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
    return {"model": stored}


@router.get("/models/{name}/file")
def download_model(name: str) -> FileResponse:
    try:
        path = STORE.resolve(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"{safe_name(name)}.cgpt",
    )


@router.delete("/models/{name}")
def delete_model(name: str) -> dict:
    try:
        STORE.delete(name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    with _CACHE_LOCK:
        _MODEL_CACHE.pop(f"{safe_name(name)}.cgpt", None)
    return {"deleted": safe_name(name)}


@router.post("/models/base/build")
def build_base_model(request: Request, body: BaseRequest | None = None) -> dict:
    _rate_limit(request, "base", 3, 3600)
    body = body or BaseRequest()
    existing = next((m for m in STORE.list_models() if m["name"] == "base"), None)
    if existing is not None:
        return {"status": "exists", "model": existing}
    job = JOBS.submit(
        "build-base",
        {"steps": body.steps, "preset": body.preset},
        _run_base_job,
        label="Building the base model",
    )
    return {"status": "queued", "job_id": job.id}


# ------------------------------------------------------------------- datasets
@router.post("/datasets")
async def upload_dataset(request: Request, files: list[UploadFile] = File(...)) -> dict:
    _rate_limit(request, "upload", 20, 3600)
    if not files:
        raise HTTPException(status_code=400, detail="no files were uploaded")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"too many files (limit {MAX_FILES})"
        )
    dataset_id = uuid.uuid4().hex[:12]
    raw_dir = DATASET_DIR / dataset_id / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict] = []
    rejected: list[dict] = []
    total_bytes = 0
    for index, upload in enumerate(files):
        filename = _safe_filename(upload.filename or f"upload-{index}.txt")
        data = await upload.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            rejected.append({"name": filename, "reason": f"larger than {MAX_FILE_MB} MB"})
            continue
        total_bytes += len(data)
        if total_bytes > MAX_TOTAL_BYTES:
            rejected.append(
                {"name": filename, "reason": f"dataset exceeds {MAX_TOTAL_MB} MB"}
            )
            break
        try:
            document = ingest.extract_bytes(data, filename)
        except ingest.IngestError as exc:
            rejected.append({"name": filename, "reason": str(exc)})
            continue
        stored_name = f"{index:02d}-{filename}"
        (raw_dir / stored_name).write_bytes(data)
        accepted.append(
            {
                "name": filename,
                "stored": stored_name,
                "bytes": len(data),
                "chars": len(document.text),
                "warnings": document.warnings,
            }
        )

    if not accepted:
        shutil.rmtree(DATASET_DIR / dataset_id, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail={
                "message": "none of the files produced usable text",
                "rejected": rejected,
            },
        )

    manifest = {
        "id": dataset_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": accepted,
        "rejected": rejected,
    }
    (DATASET_DIR / dataset_id / "dataset.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "id": dataset_id,
        "files": accepted,
        "rejected": rejected,
        "total_chars": sum(item["chars"] for item in accepted),
        "total_bytes": total_bytes,
    }


# ------------------------------------------------------------------- training
@router.post("/train")
def start_training(request: Request, body: TrainRequest) -> dict:
    _rate_limit(request, "train", 6, 3600)
    dataset_dir = DATASET_DIR / body.dataset_id / "raw"
    if not dataset_dir.is_dir():
        raise HTTPException(status_code=404, detail="dataset not found")
    parent = None
    if body.mode == "finetune":
        if not body.parent:
            raise HTTPException(
                status_code=400, detail="choose a parent model for fine-tuning"
            )
        try:
            STORE.resolve(body.parent)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="parent model not found") from exc
        parent = body.parent
    job = JOBS.submit(
        "train",
        {**body.model_dump(), "parent": parent},
        _run_train_job,
        label=f"{body.mode} run",
    )
    return {"job_id": job.id, "status": job.status}


@router.get("/train/{job_id}")
def training_status(job_id: str) -> dict:
    try:
        return JOBS.get(job_id).summary()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc


@router.get("/train/{job_id}/events")
def training_events(job_id: str) -> StreamingResponse:
    try:
        JOBS.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return StreamingResponse(
        JOBS.event_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/train/{job_id}/cancel")
def cancel_training(job_id: str) -> dict:
    try:
        job = JOBS.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="job not found") from exc
    return job.summary()


# ----------------------------------------------------------------------- chat
@router.post("/chat")
def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    _rate_limit(request, "chat", 40, 60)
    if body.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="the last message must be from the user")
    loaded = _load_model(body.model)
    messages = [message.model_dump() for message in body.messages]

    def generate():
        try:
            for event in chat_stream(
                loaded.model,
                loaded.tokenizer,
                messages,
                reasoning=body.reasoning,
                think_tokens=body.think_tokens,
                answer_tokens=body.answer_tokens,
                temperature=body.temperature,
                top_k=body.top_k,
                seed=body.seed,
            ):
                event["model"] = loaded.manifest.get("name")
                yield sse(event)
        except Exception as exc:  # noqa: BLE001 - stream errors must reach the UI
            yield sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.include_router(router)


@app.get("/api/health")
def health() -> dict:
    try:
        model_count = len(STORE.list_models())
    except Exception:
        model_count = 0
    return {
        "ok": True,
        "version": __version__,
        "torch": torch.__version__,
        "device": pipeline.pick_device(),
        "models": model_count,
        "training": any(
            job.status in ("queued", "running") for job in JOBS.jobs.values()
        ),
        "auth_required": bool(ACCESS_TOKEN),
    }


@app.get("/api/config")
def config() -> dict:
    return {
        "auth_required": bool(ACCESS_TOKEN),
        "presets": PRESETS,
        "limits": {
            "max_file_mb": MAX_FILE_MB,
            "max_files": MAX_FILES,
            "max_total_mb": MAX_TOTAL_MB,
            "max_model_mb": MAX_MODEL_MB,
        },
        "version": __version__,
    }


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    icon = WEB_DIR / "favicon.svg"
    return FileResponse(icon, media_type="image/svg+xml")
