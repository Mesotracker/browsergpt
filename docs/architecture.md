# ChainForge architecture

ChainForge is one Python process with three cooperating layers: a training core, a job queue, and an HTTP/UI layer. There is no database and no external service.

## Request-to-weights flow

```text
browser                              server process
───────                              ──────────────
POST /api/datasets  ──multipart──▶  ingest.extract_bytes()
                                      ├─ docx / pdf / html / notebook / text
                                      ├─ normalize + dedupe + cap
                                      └─ write raw uploads to data/datasets/<id>/raw
POST /api/train     ──JSON──────▶   validate → JobManager.submit()
                                      └─ worker thread:
                                           ingest.collect_documents()
                                           ByteBPE.train()          (major only)
                                           GPT(cfg)                  (or cgpt.load for finetune)
                                           write uint16 tokens.bin
                                           train_model() → callbacks
                                           cgpt.save() → store.add_file()
GET /api/train/{id}/events ──SSE──▶  backlog + live event queue
POST /api/chat      ──JSON──────▶   model cache → chat_stream() → SSE deltas
GET /api/models     ──JSON──────▶   registry scan (data/models + bundled models)
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `ModelConfig`, `TrainConfig`, presets, cosine schedule |
| `tokenizer.py` | byte-level BPE training, encoding with special tokens, JSON persistence |
| `model.py` | decoder-only GPT: pre-LN blocks, causal SDPA, GELU MLP, tied head, optimizer groups |
| `ingest.py` | mixed-format extraction, normalization, dedupe, uint16 token stream writer |
| `trainer.py` | memory-mapped batching, training loop, validation, cancellation, reports |
| `generate.py` | prompt rendering/fitting, top-k sampling, reasoning then answer phases, budgets |
| `cgpt.py` | archive write/read, allowlist validation, checksums, architecture bounds |
| `pipeline.py` | high-level train/build-base orchestration shared by CLI and server |
| `store.py` | safe names, runtime and bundled model directories, import/delete |
| `jobs.py` | bounded job history, single worker, SSE subscriber queues, cancellation |
| `server.py` | FastAPI routes, upload caps, rate limits, optional token auth, static UI |
| `cli.py` | headless parity with the UI for GitHub Actions and terminals |

## Concurrency

- One `JobManager` worker thread runs training jobs serially. A second run waits in the queue rather than oversubscribing CPU.
- Each job keeps a bounded event deque (4,000), a log deque (250), and a live subscriber list. SSE handlers replay the backlog and then relay events until a sentinel.
- Chat requests run generation in Starlette's sync-iterator threadpool. A per-model cache keeps the two most recent `LoadedModel` objects, keyed by file mtime.
- The FastAPI lifespan starts the worker. Job state is intentionally in-memory; persists only models/datasets on disk.

## Data layout

```text
data/
  models/<name>.cgpt         runtime model vault
  datasets/<id>/raw/         original uploads (provenance)
  datasets/<id>/dataset.json extraction report
  runs/<job>/                temporary token bin; removed after export
  tmp/                       import staging
models/                      bundled starter model(s), read-only
```

`CHAINFORGE_DATA` relocates everything. In Docker it defaults to `/app/data`, which is the volume to mount.

## Training pipeline

1. **Extract** each file, normalize text, reject binaries, dedupe by content hash.
2. **Tokenizer** for major runs: BPE capped at 2M characters, target vocab from the preset. Fine-tune runs reuse the parent tokenizer.
3. **Pack** every document into one uint16 stream with `<|end|>` separators; validation is the final 2% of tokens.
4. **Train** with AdamW, linear warmup (5%), cosine decay to 10% of peak, gradient clipping 1.0, eval every `min(50, steps/3)` steps.
5. **Export** with the tokenizer, architecture, lineage, compacted loss curves, and dataset manifest.

Because documents are concatenated without padding, the loss never needs an ignore index; context windows come from random offsets into the stream.

## Prompt fitting

`fit_prompt` has one hard constraint: `len(prompt) + reasoning_budget + answer_budget < context`.

1. Reserve generation budget; shrink it if the window is tiny.
2. Cap history at half of the remaining prompt room and drop the oldest tokens first.
3. Truncate the current question from the left for the rest.
4. Record every dropped token and the resulting budget in the reply metadata.

Generation then runs a reasoning phase (stopping at `<|endthink|>`, `budget`, or `<|end|>`) and an answer phase (stopping at `<|end|>` or `budget`). Every special token except the phase stop is masked, so a chat model cannot jump back into `<|user|>` mid-answer.

## Security posture

- `.cgpt` is ZIP + JSON + safetensors only; no pickle. Validation precedes allocation or model construction.
- Uploaded text is data, never code. There is no document-conversion subprocess.
- Names are sanitized; deletes are confined to the runtime model directory.
- Optional shared token gates every API route except health/config; simple in-memory per-IP limits bound chat/training/imports.
- The server never proxies model downloads from a URL; imports must be uploaded bodies.

## Known trade-offs

- BPE training is pure Python; the 2M-character cap keeps CI predictable but limits very large vocabularies.
- Generation recomputes the full context per token (no KV cache). For 256-token windows and ≤12M parameters this is simple and fast enough; it is not a serving engine.
- Training progress is in-memory, so a server restart loses the run and its queue.
- One worker means a long training job delays other jobs, by design.
