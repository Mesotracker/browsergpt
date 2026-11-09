# ChainForge

**Train a custom mini-GPT on your own documents, inspect its reasoning draft, and chat with it on the web.**

ChainForge is a Python trainer, a portable model format, and a FastAPI + vanilla-JS workshop with three tabs: **Train**, **Chat**, and **Save**. It trains a real decoder-only transformer from scratch on text you upload, or fine-tunes an existing archive, and runs comfortably on a CPU-only GitHub Actions runner.

![ChainForge](web/favicon.svg)

```text
documents ──▶ extract ──▶ byte-BPE ──▶ token stream ──▶ GPT training ──▶ .cgpt archive
                                                                    │
                                        Train / Chat / Save UI ◀────┘
```

> **Honest scope.** A 1.4M–12M parameter model trained on a small corpus is not a GPT-3.5-class assistant. It will learn your formatting, vocabulary, and short-range style, and it can imitate the reasoning/answer protocol convincingly, but it will not reliably reason about complex questions. Quality comes from more clean text and longer runs (see [Expectations](#expectations)).

---

## Browser edition (GitHub Pages)

**Live static site: <https://mesotracker.github.io/browsergpt/>**

`pages/` is a second edition of ChainForge that needs no Python, Docker, API key, Hugging Face account, or server. TensorFlow.js trains a real decoder-only transformer directly in the page, keeps models in IndexedDB, and samples chat replies on the same tab. The bundled starter is a 40,930-parameter character-level model with a 96-character context, trained on `corpus/base/`.

```bash
npm ci
npm run test:browser-model
npm run build        # assembles _site/ and bundles pages/app.js
```

The **Train** tab accepts text-like files (`.txt`, `.md`, `.csv`, `.json`, `.html`, code, notebooks), always includes the seed corpus, and supports warm-starting from the starter or a saved model. The **Chat** tab streams a visible reasoning draft and a separate answer with character budgets. The **Save** tab exports a portable `.cgpt.json` browser archive and imports it back. Everything is local to the browser; clearing site data removes saved models.

The GitHub Actions workflow `.github/workflows/pages.yml` runs the Node smoke test, builds `_site/`, enables GitHub Pages through `GITHUB_TOKEN`, and deploys it. No repository secrets are required.

> **Static-edition limits.** It is intentionally small: one decoder layer, 96-character context, and no `.docx`/`.pdf` extraction in the static build. The Python `.cgpt` ZIP format and the browser `.cgpt.json` format are not interchangeable. The starter demonstrates the pipeline and a warm-start, not a general assistant.

---

## Features

- **Real GPT training.** Pre-LN decoder-only transformer with tied embeddings, causal self-attention, GELU MLPs, AdamW, warmup + cosine decay, gradient clipping and validation loss.
- **From scratch or fine-tune.** `major` runs learn a fresh tokenizer and architecture; `fine-tune` runs continue a `.cgpt` archive with its tokenizer and context window.
- **Mixed documents.** `.txt`, `.md`, `.docx`, `.pdf` (text layer), `.csv`, `.json`, `.html`, YAML, source code, notebooks, and any readable text file. Scanned PDFs and binary media are rejected with a clear reason instead of silently ignored.
- **Custom amount of steps.** Any integer from 1 to 20,000 in the UI; presets are only shortcuts.
- **Portable `.cgpt` format.** ZIP + JSON + safetensors, no pickle, checksummed on import, allowlisted members, bounded architecture.
- **Visible reasoning and token budget.** A scratchpad protocol (`<|think|> … <|endthink|>`) streamed before the answer, with prompt/reasoning/answer/context accounting on every reply.
- **Blue-black-gold 3D UI.** Beveled panels, hard-edged 3D buttons, live loss chart, streaming transcript.
- **Base model builder.** One click or one workflow input builds the starter base model from the seed corpus.
- **GitHub Actions → web.** CI tests, a base-model training workflow with artifacts, and a Docker image published to GHCR with optional Hugging Face Spaces deployment.
- **Safe by default.** No pickle, upload caps, archive validation, optional access token, per-IP rate limits, no code execution from documents.

---

## Quickstart

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -e ".[dev]"

# Optional: build the starter base model from corpus/base (~3-6 min on CPU)
python -m chainforge build-base --steps 300 --out models/base.cgpt

python -m chainforge serve --port 8000
```

Open <http://localhost:8000>. The **Train** tab uploads documents, extracts text, trains, and archives the result. The **Chat** tab streams replies from your local weights. The **Save** tab downloads and imports `.cgpt` files.

If you skip the base-model build, the UI's **Build base model** button does the same thing on the server, and GitHub Actions can build and commit it for you.

### Docker

```bash
docker build -t chainforge .
docker run --rm -p 8000:7860 -v chainforge-data:/app/data chainforge
# open http://localhost:8000
```

The image defaults to port `7860` (Hugging Face Spaces convention); set `PORT=8000` to change it.

---

## Web UI

### Train

1. Drop in any number of documents (up to 24 files, 25 MB each, 150 MB total by default).
2. Choose **Major** (fresh tokenizer + model) or **Fine-tune** (pick a parent model).
3. Set steps, batch size and learning rate. The **Run monitor** shows step, train loss, validation loss, tokens/s, ETA, a gold loss curve, and a live log.
4. When the run finishes, the model is archived and offered as the next chat model.

### Chat

- Choose the model, toggle **Visible reasoning**, and set the reasoning and answer token budgets.
- The transcript shows a collapsible **reasoning draft** block, followed by the answer.
- The **Context meter** breaks down prompt, reasoning, answer, and context, and names the stop reason (`endthink`, `end`, `budget`, or `off`).
- Temperature and top-k are exposed; generation is token-by-token over SSE.

### Save

- **Model vault** with parameters, steps, context, loss, creation date, lineage and size.
- Download any archive; import a `.cgpt` to chat with it or continue training it.
- Bundled models are read-only; user models can be deleted.
- Optional server access token is stored only in the browser's `localStorage`.

### Mobile

Below 700px the tab bar becomes a fixed bottom navigation with safe-area padding, the conversation is ordered before its settings, and controls use 44px touch targets. Chat and form fields switch to 16px text to avoid iOS focus zoom. The responsive layout is checked at 360×800, 390×844, and 768×1024.

---

## Training modes

| | Major | Fine-tune |
|---|---|---|
| Initial weights | random | parent `.cgpt` |
| Tokenizer | learned from the new dataset | reused from parent |
| Architecture | preset (`base`/`wide`) + overrides | locked to parent |
| Default learning rate | `4e-4` | `1e-4` |
| Typical use | new domain/model | adapt an existing model |

| Preset | Layers | Heads | d_model | Context | Vocab | Params |
|---|---|---|---|---|---|---|
| `base` | 4 | 4 | 160 | 256 | ≤1024 | ~1.4M |
| `wide` | 6 | 6 | 384 | 384 | ≤4096 | ~12.3M |

Steps are user-controlled. Training packs documents into one token stream with `<|end|>` separators and takes a 98/2 train/validation split.

### CLI

```bash
# Train from files and/or directories (recursive)
python -m chainforge train notes/ papers/ README.md \
  --out models/domain.cgpt --steps 500

# Fine-tune an imported archive
python -m chainforge train notes/ --mode finetune \
  --init models/base.cgpt --out models/notes-ft.cgpt \
  --steps 250 --lr 0.0001

# Build the starter model
python -m chainforge build-base --steps 300 --out models/base.cgpt

# Inspect / download metadata
python -m chainforge inspect models/base.cgpt

# Terminal chat with the reasoning draft
python -m chainforge chat --model models/base.cgpt \
  --prompt "What is ChainForge?" --think 48 --answer 96

# Verify train → export → import → chat without touching your data
python -m chainforge smoke --steps 8

# Serve the UI
python -m chainforge serve --host 0.0.0.0 --port 8000
```

Progress goes to stderr; `--json` prints a machine-readable summary on stdout, which is how the GitHub Actions workflow captures run stats.

---

## Data support

| Input | Handling |
|---|---|
| `.txt .md .rst .log .csv .tsv .json .jsonl .yaml .yml .toml .ini .py .js .ts .html .xml .css .sql .sh` and source code | Decoded as text; HTML/XML tags stripped |
| `.docx` | Paragraphs and table cells via `python-docx` |
| `.pdf` | Text layer via `pypdf`; scanned PDFs need OCR first and are rejected |
| `.ipynb` | Markdown and code cells |
| Unknown extension | Decoded as UTF-8/CP1252 with a printable-ratio check |
| `.doc .xls .xlsx .pptx` | Rejected with guidance (convert to `.docx`/CSV/text) |
| Images, audio, video, archives, model binaries | Rejected (no OCR/transcription in this build) |

Text is normalized (NFC, newlines, whitespace), capped at 8M characters per file, and near-duplicate documents are removed by content hash.

---

## `.cgpt` format

A `.cgpt` file is a ZIP archive containing exactly:

| Member | Contents |
|---|---|
| `manifest.json` | format version, id, name, created, architecture, parameter count, tokenizer summary, weights SHA-256, lineage |
| `tokenizer.json` | byte-level BPE tokens, merges, special chat tokens |
| `weights.safetensors` | float32 tensors (no pickle) |
| `training.json` | steps, hyperparameters, loss curves, dataset manifest, device and timing |

Import rules: member allowlist, no directories, no absolute/`..` paths, per-member and total size caps, compression-ratio sanity, checksum verification, architecture bounds, and strict state-dict matching (with an allowed tied-embedding exception). See [docs/cgpt-format.md](docs/cgpt-format.md).

Fine-tuning preserves the parent tokenizer and architecture, records `lineage.parent` and `lineage.parent_id`, and exports a complete new archive.

---

## Visible reasoning and token budget

The chat protocol is:

```text
<|bos|><|user|>
question
<|assistant|>
<|think|>
scratchpad notes
<|endthink|>
answer
<|end|>
```

Generation runs in two phases. The reasoning phase is capped by the reasoning budget and stops on `<|endthink|>`; if it hits the budget the server appends the marker itself. The answer phase is capped separately and stops on `<|end|>`. All special tokens except the phase stop token are masked during sampling.

`fit_prompt` enforces the hard context limit: it reserves generation budget, keeps at most half the remaining prompt room for history, trims the oldest history first, then truncates the current question from the left, and reports how many tokens were dropped. The UI shows the split and every stop reason.

**The trace is a draft the model wrote to condition its answer.** It is not a faithful record of hidden computation, and small models often produce shallow or wrong notes. The UI says this explicitly instead of presenting it as verified chain-of-thought.

---

## GitHub Actions

### `ci.yml`

On pushes and pull requests: installs CPU PyTorch, runs the pytest suite (tokenizer, ingestion, `.cgpt` safety, training, generation budgets, full API flow), then runs `chainforge smoke`.

### `train-base.yml` (manual)

Inputs: `steps`, `preset` (`base`/`wide`), `commit_model`. It trains `models/base.cgpt`, uploads it plus `base-summary.json` as an artifact, and — when `commit_model` is true — commits the archive back to the branch. Run it from **Actions → Train base model → Run workflow**.

### `pages.yml`

On pushes to `main` or manual dispatch: installs the Node static-site dependencies, runs the browser-transformer smoke test, builds `_site/`, and deploys the static edition to GitHub Pages. `actions/configure-pages@v5` runs with `enablement: true`, so the repository's Pages site is configured without a manual settings step. The output is a relative-path site that works at `/browsergpt/`.

### `deploy.yml`

On pushes to `main` (or a manual **Run workflow**): builds the Docker image and pushes `ghcr.io/<owner>/<repo>:latest` and a short-SHA tag. If `HF_TOKEN` and the `HF_SPACE` repository variable are configured, it also publishes the app to a Hugging Face Space and overwrites the Space README with the required Docker metadata. Without those two values the workflow explicitly reports that web deployment was skipped; the image is still published.

Hugging Face Spaces' free CPU tier is the realistic zero-cost target for this stack (the image is large because of PyTorch, and free container hosts with persistent disks are scarce). Any Docker host works: the app is one process, one port, and a writable data directory. Attach a volume to `/app/data` if you want trained models to survive restarts.

### Deployment configuration

| Name | Kind | Purpose |
|---|---|---|
| `HF_TOKEN` | secret | Hugging Face write token for Spaces |
| `HF_SPACE` | repository variable | e.g. `your-user/chainforge` |
| `CHAINFORGE_TOKEN` | env | Optional shared access token for all `/api` routes |
| `CHAINFORGE_ROOT` | env | Project root holding `web/`, `corpus/`, `models/` (default: cwd if it looks like a checkout; `/app` in Docker) |
| `CHAINFORGE_DATA` | env | Data dir (default `./data`; `/app/data` in Docker) |
| `CHAINFORGE_MAX_UPLOAD_MB` | env | Per-file upload cap (default 25) |
| `CHAINFORGE_MAX_FILES` | env | Files per dataset (default 24) |
| `CHAINFORGE_MAX_TOTAL_UPLOAD_MB` | env | Dataset cap (default 150) |
| `CHAINFORGE_MAX_MODEL_MB` | env | `.cgpt` import cap (default 256) |
| `PORT` | env | Container port (default 7860) |

The app has no cloud dependency, no telemetry, and sends no document text anywhere. Training happens in the server process.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Public status: device, torch version, model count |
| `GET` | `/api/config` | Public limits, presets, auth flag |
| `GET` | `/api/models` | Model vault |
| `POST` | `/api/models/import` | Import and validate a `.cgpt` |
| `GET` | `/api/models/{name}/file` | Download a model |
| `DELETE` | `/api/models/{name}` | Delete a user model |
| `POST` | `/api/models/base/build` | Queue a base-model build |
| `POST` | `/api/datasets` | Multipart upload + extraction report |
| `POST` | `/api/train` | Queue a major/fine-tune run |
| `GET` | `/api/train/{id}` | Job status, progress and logs |
| `GET` | `/api/train/{id}/events` | SSE progress stream |
| `POST` | `/api/train/{id}/cancel` | Request cancellation |
| `POST` | `/api/chat` | SSE reasoning/answer stream with budget |

All `/api` routes except health/config require `X-Chainforge-Token` when `CHAINFORGE_TOKEN` is set. Training jobs run on a single in-process worker so concurrent uploads cannot exhaust CPU.

---

## Limits and security

- Uploads: 24 files, 25 MB each, 150 MB per dataset by default; 8M extracted characters per file.
- Models: archives up to 512 MB, architectures estimated at ≤ 60M parameters, strict allowlist.
- No pickle loading, no `exec`, no shelling out for document conversion.
- Chat rate limit 40/min/IP; training 6/hour/IP; imports 10/hour/IP; base builds 3/hour/IP.
- Training runs are capped at 20,000 steps from the UI/API.
- Job history is bounded in memory; datasets and models live on disk and are not deleted automatically.
- Hosted storage may be ephemeral. Download archives you want to keep.
- The static browser edition reads text-like files only; `.docx` and `.pdf` extraction is available in the Python edition. Its browser archives are JSON and are separate from the Python `.cgpt` format.

---

## Expectations

The bundled corpus is a small seed set for demonstrating the pipeline. A fresh base build reliably learns the chat/think formatting and the project vocabulary; it does not become a general assistant.

To move toward stronger output:

1. **Add data.** Hundreds of thousands to millions of characters of clean, consistent text beat any hyperparameter tweak.
2. **Use the `wide` preset** (or `--ctx`, `--vocab-size` overrides) when you have CPU time or a GPU.
3. **Train longer**, but watch validation loss; a rising val loss means memorization.
4. **Fine-tune repeatedly** on curated examples; lineage is preserved in every archive.
5. **Expect scale limits.** GPT-3.5-class behavior needs orders of magnitude more parameters and data than a CI-friendly CPU run.

---

## Tests

```bash
python -m pytest -q
python -m chainforge smoke --steps 8

# Browser transformer: forward pass, loss improvement, sampling, and JSON round-trip
npm run test:browser-model
```

The Python suite covers unicode/special-token roundtrips, deterministic BPE, docx/text/unknown/binary ingestion, archive fidelity and logits equality after export/import, checksum and traversal rejection, training/cancellation, prompt fitting and budgets, and a full API flow (import → dataset → train → list → download → chat). The browser smoke test exercises the shared TensorFlow.js engine in Node.

---

## Project layout

```text
chainforge/
  config.py      presets, architecture/training configs, LR schedule
  tokenizer.py   byte-level BPE with chat special tokens
  model.py       GPT (pre-LN, tied embeddings, causal attention)
  ingest.py      mixed-document extraction and token streams
  trainer.py     batching, training loop, evaluation, cancellation
  generate.py    sampling, prompt fitting, reasoning stream, budgets
  cgpt.py        .cgpt archive read/write and validation
  pipeline.py    dataset → train → export orchestration
  store.py       runtime/bundled model registry
  jobs.py        single-worker queue with SSE fan-out
  server.py      FastAPI application and API surface
  cli.py         train/build-base/inspect/chat/smoke/serve
corpus/base/     seed documents for the starter model
docs/            architecture and format notes
web/             Python edition UI (index.html, app.css, app.js)
pages/           static browser edition (index.html, app.js, engine.js, base-model.json)
scripts/         build_pages.py, browser base trainer, browser smoke test
tests/           pytest suite
.github/         CI, base-model workflow, container deploy, Pages deploy
```

## License

MIT.
