# The `.cgpt` container

`.cgpt` is ChainForge's portable model format: a ZIP archive with JSON metadata, a JSON tokenizer, and safetensors weights. It is designed to be inspectable, safe to import, and sufficient to resume training.

## Layout

```text
my-model.cgpt (ZIP, deflate)
├── manifest.json         required
├── tokenizer.json        required
├── weights.safetensors   required
└── training.json         optional
```

No other members are allowed. No pickle is used at any point.

## `manifest.json`

```json
{
  "format": "cgpt/1.0",
  "id": "9f0d7e6c5b4a3210...",
  "name": "notes-ft",
  "created": "2026-11-09T12:00:00Z",
  "arch": {
    "vocab_size": 1024,
    "ctx": 256,
    "d_model": 160,
    "n_layer": 4,
    "n_head": 4,
    "dropout": 0.1,
    "tie_embeddings": true
  },
  "params": 1435200,
  "tokenizer": {
    "type": "byte_bpe",
    "vocab_size": 1024,
    "specials": ["<|assistant|>", "<|bos|>", "<|end|>", "..."]
  },
  "weights": {
    "path": "weights.safetensors",
    "sha256": "…",
    "dtype": "float32"
  },
  "lineage": {
    "mode": "finetune",
    "parent": "base",
    "parent_id": "…"
  },
  "caps": {
    "reasoning_protocol": true,
    "chat_protocol": "user/assistant/think/endthink/end"
  }
}
```

`format` is versioned. Anything whose prefix is not `cgpt` is rejected; a future `cgpt/2.0` can introduce new optional members while old readers keep rejecting unknown names by design.

## `tokenizer.json`

```json
{
  "type": "byte_bpe",
  "version": 1,
  "tokens": ["… 256 byte tokens …", "learned mergers …", "<|pad|>", "<|bos|>", "…"],
  "merges": [[104, 101], [256, 108], "…"],
  "specials": {"<|pad|>": 1024, "<|bos|>": 1025, "…": 0}
}
```

IDs are positional: 0–255 are single bytes, learned merges come next in order, specials last. Because all 256 bytes are in the alphabet, any UTF-8 document is encodable — there is no unknown-token path, even for text the tokenizer never saw. `<|end|>` separates documents in the training stream.

## `weights.safetensors`

Tensor names match `GPT.state_dict()`. When embeddings are tied, `head.weight` is omitted; on load it is re-tied to `tok_emb.weight`. Tensors are exported as contiguous float32 on CPU. The manifest stores the SHA-256 of the raw safetensors bytes.

## `training.json`

Free-form but JSON-safe run metadata: mode, steps, batch size, learning rate, context, seed, device, seconds, tokens seen, initial/final/best validation loss, compacted loss curves, and a dataset manifest with per-file character/token counts and rejected files.

## Import validation

`cgpt.load` performs, in order:

1. File size ≤ 512 MB; valid ZIP.
2. Member names are in the allowlist, unique, non-directories, and free of traversal.
3. Per-member uncompressed size ≤ 400 MB; total ≤ 1 GB; compression ratio < 400:1.
4. `manifest.json` parses and its format prefix is `cgpt`.
5. Architecture parses under `ModelConfig` bounds: vocab 256–65,536, context 32–2,048, d_model 16–2,048 and divisible by heads, ≤ 48 layers, ≤ 64 heads, dropout < 1.
6. Estimated parameter count ≤ 60M.
7. Tokenizer parses; vocab size equals `arch.vocab_size`.
8. Weights SHA-256 matches.
9. Safetensors tensors load; state-dict keys match exactly, with `head.weight` allowed to be absent only when embeddings are tied.
10. The model is placed in eval mode on the requested device.

`read_manifest` runs the container checks (1–6) without building the model, which is what the vault listing uses.

## Import → retrain

Fine-tuning resolves the parent archive, loads tokenizer + architecture, trains with a lower learning rate, and writes a new archive whose `lineage.parent`/`parent_id` points at the source. Imports may be chained indefinitely; each archive is self-contained.

## Reproducing a model

The archive contains everything needed for inference and further training except the original documents. `training.json` lists the dataset sizes and file names, not their text. Keep your source documents if you need bit-for-bit reproducibility of tokenization.
