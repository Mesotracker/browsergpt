"""In-process training job queue with SSE fan-out."""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Callable, Iterator

TERMINAL = {"done", "error", "cancelled"}
_SENTINEL = object()


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


class Job:
    def __init__(self, kind: str, request: dict, runner: Callable[["Job"], dict]) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.request = request
        self.runner = runner
        self.status = "queued"
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.events: deque[dict] = deque(maxlen=4000)
        self.logs: deque[str] = deque(maxlen=250)
        self.progress: dict = {}
        self.result: dict | None = None
        self.error: str | None = None
        self.cancel_event = threading.Event()
        self.subscribers: list[queue.Queue] = []
        self.lock = threading.Lock()

    def emit(self, event: dict) -> None:
        event = dict(event)
        event.setdefault("job_id", self.id)
        with self.lock:
            self.events.append(event)
            if event.get("type") == "log" and event.get("message"):
                self.logs.append(str(event["message"]))
            if event.get("type") == "step":
                self.progress = {
                    "step": event.get("step"),
                    "total": event.get("total"),
                    "loss": event.get("loss"),
                    "lr": event.get("lr"),
                    "tokens": event.get("tokens"),
                    "tok_s": event.get("tok_s"),
                    "eta_s": event.get("eta_s"),
                }
            elif event.get("type") == "eval":
                self.progress = {
                    **self.progress,
                    "val_loss": event.get("val_loss"),
                    "best_val_loss": event.get("best_val_loss"),
                }
            for subscriber in self.subscribers:
                subscriber.put(event)

    def finish(self, status: str, *, result: dict | None = None, error: str | None = None) -> None:
        self.status = status
        self.finished = time.time()
        self.result = result
        self.error = error
        self.emit(
            {
                "type": "status",
                "status": status,
                "result": result,
                "error": error,
            }
        )
        with self.lock:
            for subscriber in self.subscribers:
                subscriber.put(_SENTINEL)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "progress": self.progress,
            "logs": list(self.logs),
            "result": self.result,
            "error": self.error,
            "request": {
                k: v for k, v in self.request.items() if k not in ("token", "password")
            },
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.order: deque[str] = deque(maxlen=100)
        self.lock = threading.Lock()
        self.queue: queue.Queue[Job] = queue.Queue()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._worker, name="chainforge-jobs", daemon=True)
        self.thread.start()

    def submit(
        self, kind: str, request: dict, runner: Callable[[Job], dict], label: str = ""
    ) -> Job:
        job = Job(kind, request, runner)
        with self.lock:
            self.jobs[job.id] = job
            self.order.append(job.id)
            stale = [key for key in self.jobs if key not in self.order]
            for key in stale:
                del self.jobs[key]
        job.emit({"type": "queued", "label": label, "position": self.queue.qsize() + 1})
        self.queue.put(job)
        return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel_event.set()
        if job.status == "queued":
            job.finish("cancelled", error="cancelled before start")
        return job

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            if job.cancel_event.is_set():
                if job.status not in TERMINAL:
                    job.finish("cancelled", error="cancelled before start")
                continue
            job.status = "running"
            job.started = time.time()
            job.emit({"type": "start", "status": "running"})
            try:
                result = job.runner(job)
                if job.cancel_event.is_set():
                    job.finish("cancelled", error="cancelled by user")
                else:
                    job.finish("done", result=result)
            except Exception as exc:  # noqa: BLE001 - jobs must never kill the worker
                traceback.print_exc()
                if job.cancel_event.is_set():
                    job.finish("cancelled", error="cancelled by user")
                else:
                    job.finish("error", error=str(exc))

    def event_stream(self, job_id: str) -> Iterator[str]:
        job = self.get(job_id)
        subscriber: queue.Queue = queue.Queue()
        with job.lock:
            backlog = list(job.events)
            terminal = job.status in TERMINAL
            if not terminal:
                job.subscribers.append(subscriber)
        try:
            for event in backlog:
                yield sse(event)
            if terminal:
                return
            while True:
                try:
                    item = subscriber.get(timeout=10)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is _SENTINEL:
                    return
                yield sse(item)  # type: ignore[arg-type]
        finally:
            with job.lock:
                if subscriber in job.subscribers:
                    job.subscribers.remove(subscriber)
