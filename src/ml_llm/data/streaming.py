from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _format_turns(turns: list[dict]) -> str | None:
    parts = []
    for turn in turns:
        content = turn.get("content", "").strip()
        if not content:
            continue
        role = "<user>" if turn.get("role") == "user" else "<assistant>"
        parts.append(f"{role} {content}")
    if not parts:
        return None
    return "<bos> " + " ".join(parts) + " <eos>"


def sample_texts(dataset_dir: Path, *, limit: int = 50_000) -> list[str]:
    if not dataset_dir.exists():
        return []

    texts = []

    for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
        with jsonl_path.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                formatted = _format_turns(row.get("turns") or [])
                if formatted:
                    texts.append(formatted)

                if limit and len(texts) >= limit:
                    break

    return texts[:limit]


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _dataset_signature(dataset_dir: Path, context_len: int) -> str:
    h = hashlib.sha256()
    h.update(str(context_len).encode())
    h.update(b"fmt3")
    for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
        stat = jsonl_path.stat()
        h.update(jsonl_path.name.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()


def build_examples_cache(
    dataset_dir: Path,
    tokenizer,
    context_len: int,
    cache_path: Path,
    *,
    max_examples: int | None = None,
    force: bool = False,
) -> int:
    signature = _dataset_signature(dataset_dir, context_len)
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

    total_tokens = 0
    seen = 0
    buffer: list[int] = []
    done = False

    with open(cache_path, "wb", buffering=1024 * 1024) as out_fh:
        for jsonl_path in sorted(dataset_dir.glob("*.jsonl")):
            if done:
                break
            print(f"Encoding: {jsonl_path.name}", flush=True)
            with jsonl_path.open() as fh:
                for line in fh:
                    seen += 1
                    if seen % 5000 == 0:
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
                        done = True
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
