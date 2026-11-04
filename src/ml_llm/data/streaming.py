from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

from ml_llm.data.tokenizer import strip_specials

FILE_WEIGHTS = {
    "personachat_converted": 4,
    "cornell_converted": 3,
    "dailydialog": 3,
    "wildchat_filtered": 2,
    "sharegpt_filtered": 1,
    "corpora": 1,
    "lamini_filtered": 1,
}
FILE_CAPS = {"lamini_filtered": 150_000}


def _format_turns(turns: list[dict]) -> str | None:
    parts = []
    for turn in turns:
        content = strip_specials(turn.get("content") or "").strip()
        if not content:
            continue
        role = "<user>" if turn.get("role") == "user" else "<assistant>"
        parts.append(f"{role} {content}")
    if not parts:
        return None
    return "<bos> " + " ".join(parts) + " <eos>"


def sample_texts(dataset_dir: Path, *, limit: int = 50_000, seed: int = 0) -> list[str]:
    if not dataset_dir.exists():
        return []

    files = sorted(dataset_dir.glob("*.jsonl"))
    if not files:
        return []

    per_file = max(1, limit // len(files)) if limit else None
    rng = random.Random(seed)
    texts: list[str] = []

    for jsonl_path in files:
        reservoir: list[str] = []
        seen = 0
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                seen += 1
                if per_file is None or len(reservoir) < per_file:
                    reservoir.append(line)
                else:
                    j = rng.randrange(seen)
                    if j < per_file:
                        reservoir[j] = line

        for line in reservoir:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            formatted = _format_turns(row.get("turns") or [])
            if formatted:
                texts.append(formatted)

    return texts


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _dataset_signature(dataset_dir: Path, context_len: int, tokenizer_fp: str) -> str:
    h = hashlib.sha256()
    h.update(str(context_len).encode())
    h.update(b"fmt5")
    h.update(json.dumps([FILE_WEIGHTS, FILE_CAPS], sort_keys=True).encode())
    h.update(tokenizer_fp.encode())
    for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
        stat = jsonl_path.stat()
        h.update(jsonl_path.name.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()


def _interleaved_lines(dataset_dir: Path):
    files = sorted(dataset_dir.glob("*.jsonl"))
    handles = {p: p.open(encoding="utf-8") for p in files}
    counts = {p: 0 for p in files}
    live = list(files)
    try:
        while live:
            for p in list(live):
                weight = FILE_WEIGHTS.get(p.stem, 1)
                cap = FILE_CAPS.get(p.stem)
                for _ in range(weight):
                    if cap is not None and counts[p] >= cap:
                        live.remove(p)
                        break
                    line = handles[p].readline()
                    if not line:
                        live.remove(p)
                        break
                    counts[p] += 1
                    yield line
    finally:
        for handle in handles.values():
            handle.close()


def build_examples_cache(
    dataset_dir: Path,
    tokenizer,
    context_len: int,
    cache_path: Path,
    *,
    max_examples: int | None = None,
    force: bool = False,
) -> int:
    signature = _dataset_signature(dataset_dir, context_len, tokenizer.fingerprint())
    meta_file = _meta_path(cache_path)

    if not force and cache_path.exists() and meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError:
            meta = {}
        if meta.get("signature") == signature and meta.get("context_len") == context_len:
            print(f"Using cached dataset ({meta.get('num_rows', 0):,} rows)", flush=True)
            return meta.get("num_rows", 0)

    dtype = np.uint16 if tokenizer.vocab_size < 65536 else np.uint32
    dtype_name = np.dtype(dtype).name
    token_limit = max_examples * context_len if max_examples else None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_file.unlink(missing_ok=True)

    total_tokens = 0
    seen = 0
    buffer: list[int] = []

    with open(cache_path, "wb", buffering=1024 * 1024) as out_fh:
        print("Encoding interleaved dataset", flush=True)
        for line in _interleaved_lines(dataset_dir):
            seen += 1
            if seen % 20000 == 0:
                print(f"  {seen:,} lines, {total_tokens:,} tokens", flush=True)
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            formatted = _format_turns(row.get("turns") or [])
            if not formatted:
                continue

            buffer.extend(tokenizer.encode(formatted))

            if len(buffer) >= 1_000_000:
                out_fh.write(np.asarray(buffer, dtype=dtype).tobytes())
                total_tokens += len(buffer)
                buffer = []

            if token_limit and total_tokens >= token_limit:
                break

        if buffer:
            out_fh.write(np.asarray(buffer, dtype=dtype).tobytes())
            total_tokens += len(buffer)

        out_fh.flush()
        os.fsync(out_fh.fileno())

    num_rows = total_tokens // context_len
    if max_examples:
        num_rows = min(num_rows, max_examples)

    if num_rows == 0:
        cache_path.write_bytes(b"")
        return 0

    meta_file.write_text(
        json.dumps({
            "dtype": dtype_name,
            "num_rows": num_rows,
            "context_len": context_len,
            "signature": signature,
        })
    )

    return num_rows


class MemmapConversationDataset:
    def __init__(self, cache_path: Path, context_len: int, num_rows: int):
        if num_rows == 0 or not cache_path.exists() or cache_path.stat().st_size == 0:
            self._data = np.zeros((0, context_len), dtype=np.uint16)
            self._num_rows = 0
            return

        dtype = np.uint16
        meta_file = _meta_path(cache_path)
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                dtype = np.dtype(meta["dtype"])
            except (json.JSONDecodeError, KeyError, TypeError):
                dtype = np.uint16

        self._data = np.memmap(
            cache_path,
            dtype=dtype,
            mode="r",
            shape=(num_rows, context_len),
        )
        self._num_rows = num_rows

    def __len__(self) -> int:
        return self._num_rows

    def __getitem__(self, index: int):
        row = self._data[index]
        x = row[:-1].copy().astype(np.int64)
        y = row[1:].copy().astype(np.int64)
        y[y == 0] = -1
        return x, y
