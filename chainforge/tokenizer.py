"""A small, dependency-free byte-level BPE tokenizer.

The vocabulary layout is stable and intentional:

* ids 0-255               single bytes (GPT-2 byte/unicode mapping)
* ids 256..256+merges-1   learned BPE merges
* final ids               special chat tokens

Because the base alphabet is all 256 bytes, any file can be encoded without an
"unknown token" path, including text the tokenizer was never trained on.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

DEFAULT_SPECIALS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|user|>",
    "<|assistant|>",
    "<|think|>",
    "<|endthink|>",
    "<|end|>",
)

# GPT-2 style pre-tokenizer. BPE is only ever learned inside these chunks.
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode() -> dict[int, str]:
    """Reversible mapping from bytes to printable unicode code points."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {ch: b for b, ch in BYTE_ENCODER.items()}


def _replace_pair(word: Sequence[int], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    n = len(word)
    while i < n:
        if i + 1 < n and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class ByteBPE:
    """Learned byte-level byte-pair encoding with chat special tokens."""

    def __init__(
        self,
        tokens: list[str],
        merges: list[tuple[int, int]],
        specials: dict[str, int] | None = None,
    ) -> None:
        if len(tokens) < 256:
            raise ValueError("tokenizer vocab must contain at least the 256 byte tokens")
        self.tokens = list(tokens)
        self.merges = list(merges)
        self.specials: dict[str, int] = dict(specials or {})
        self._token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(self.tokens)}
        self._id_to_special: dict[int, str] = {i: s for s, i in self.specials.items()}
        self._merge_ranks: dict[tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._merge_id: dict[tuple[int, int], int] = {
            pair: 256 + rank for rank, pair in enumerate(self.merges)
        }
        self._word_cache: dict[str, list[int]] = {}
        if self.specials:
            pattern = "|".join(
                re.escape(s) for s in sorted(self.specials, key=len, reverse=True)
            )
            self._special_re: re.Pattern[str] | None = re.compile(pattern)
        else:
            self._special_re = None

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 1024,
        specials: Sequence[str] = DEFAULT_SPECIALS,
        max_chars: int = 2_000_000,
    ) -> "ByteBPE":
        """Learn merges from ``texts``.

        Training is capped at ``max_chars`` characters so a large upload can
        never turn into an unbounded CPU job.
        """
        if vocab_size < 256 + len(specials) + 1:
            raise ValueError(
                f"vocab_size must be at least {256 + len(specials) + 1} for this tokenizer"
            )
        text = "\n".join(t for t in texts if t)
        if len(text) > max_chars:
            text = text[:max_chars]

        # Specials are atomic, so strip them before learning merges.
        if specials:
            special_re = re.compile(
                "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True))
            )
            text = special_re.sub(" ", text)

        word_freq: Counter[tuple[int, ...]] = Counter()
        for match in _PAT.finditer(text):
            encoded = match.group().encode("utf-8")
            if encoded:
                word_freq[tuple(encoded)] += 1

        tokens: list[str] = [BYTE_ENCODER[b] for b in range(256)]
        merges: list[tuple[int, int]] = []
        target_merges = vocab_size - 256 - len(specials)

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
        for word, freq in word_freq.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                pair_words[pair].add(word)

        while len(merges) < target_merges and pair_counts:
            pair, _count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            new_id = len(tokens)
            merged_token = tokens[pair[0]] + tokens[pair[1]]
            tokens.append(merged_token)
            merges.append(pair)

            affected = list(pair_words.get(pair, ()))
            for word in affected:
                freq = word_freq.pop(word, 0)
                if not freq:
                    continue
                for p in zip(word, word[1:]):
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    bucket = pair_words.get(p)
                    if bucket is not None:
                        bucket.discard(word)
                new_word = _replace_pair(word, pair, new_id)
                word_freq[new_word] += freq
                for p in zip(new_word, new_word[1:]):
                    pair_counts[p] += freq
                    pair_words[p].add(new_word)

            pair_counts.pop(pair, None)
            pair_words.pop(pair, None)
            if len(pair_words) > 400_000:
                pair_words = defaultdict(
                    set, {p: w for p, w in pair_words.items() if w}
                )

        vocab = tokens + list(specials)
        if len(vocab) > 65536:
            raise ValueError("vocab_size too large")
        tokenizer = cls(vocab, merges, {s: len(tokens) + i for i, s in enumerate(specials)})
        return tokenizer

    # ----------------------------------------------------------------- encode
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def special_id(self, token: str) -> int:
        try:
            return self.specials[token]
        except KeyError as exc:
            raise KeyError(f"{token!r} is not a special token of this tokenizer") from exc

    def _merge_word(self, ids: list[int]) -> list[int]:
        if len(ids) < 2:
            return ids
        ids = list(ids)
        while len(ids) > 1:
            best_rank: int | None = None
            best_pos = -1
            for i in range(len(ids) - 1):
                rank = self._merge_ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pos = i
            if best_pos < 0:
                break
            pair = (ids[best_pos], ids[best_pos + 1])
            ids[best_pos : best_pos + 2] = [self._merge_id[pair]]
        return ids

    def _encode_word(self, word: str) -> list[int]:
        chars = [BYTE_ENCODER[b] for b in word.encode("utf-8", errors="replace")]
        ids = [self._token_to_id[c] for c in chars]
        return self._merge_word(ids)

    def encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        cache = self._word_cache
        for match in _PAT.finditer(text):
            word = match.group()
            cached = cache.get(word)
            if cached is None:
                cached = self._encode_word(word)
                if len(cache) < 200_000:
                    cache[word] = cached
            out.extend(cached)
        return out

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special or not self._special_re:
            return self.encode_ordinary(text)
        out: list[int] = []
        pos = 0
        for match in self._special_re.finditer(text):
            if match.start() > pos:
                out.extend(self.encode_ordinary(text[pos : match.start()]))
            out.append(self.specials[match.group()])
            pos = match.end()
        if pos < len(text):
            out.extend(self.encode_ordinary(text[pos:]))
        return out

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        pieces: list[str] = []
        buf: list[str] = []
        for token_id in ids:
            special = self._id_to_special.get(int(token_id))
            if special is not None:
                if buf:
                    pieces.append(self._bytes_to_text(buf))
                    buf = []
                if not skip_special:
                    pieces.append(special)
            else:
                if not 0 <= int(token_id) < len(self.tokens):
                    raise ValueError(f"token id out of range: {token_id}")
                buf.append(self.tokens[int(token_id)])
        if buf:
            pieces.append(self._bytes_to_text(buf))
        return "".join(pieces)

    @staticmethod
    def _bytes_to_text(parts: list[str]) -> str:
        # BPE tokens can be multi-byte strings; flatten before mapping back.
        data = bytes(BYTE_DECODER[c] for c in "".join(parts))
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "type": "byte_bpe",
            "version": 1,
            "tokens": self.tokens,
            "merges": [[a, b] for a, b in self.merges],
            "specials": self.specials,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ByteBPE":
        if data.get("type") != "byte_bpe":
            raise ValueError("unsupported tokenizer type")
        tokens = data.get("tokens")
        merges = data.get("merges")
        specials = data.get("specials", {})
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError("invalid tokenizer vocabulary")
        if not isinstance(merges, list):
            raise ValueError("invalid tokenizer merges")
        parsed_merges = []
        for pair in merges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(x, int) for x in pair)
            ):
                raise ValueError("invalid tokenizer merge entry")
            parsed_merges.append((pair[0], pair[1]))
        if not isinstance(specials, dict) or not all(
            isinstance(k, str) and isinstance(v, int) for k, v in specials.items()
        ):
            raise ValueError("invalid tokenizer special tokens")
        return cls(tokens, parsed_merges, specials)
