from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]
SPECIAL_SET = set(SPECIAL_TOKENS)
_SPECIAL_NAMES = "|".join(t[1:-1] for t in SPECIAL_TOKENS)
SPECIAL_RE = re.compile(rf"\s*(<(?:{_SPECIAL_NAMES})>)\s*")
SPECIAL_STRIP_RE = re.compile(rf"<(?:{_SPECIAL_NAMES})>")
PIECE_RE = re.compile(r" ?\w+| ?[^\w\s]|\s+(?!\S)|\s+")
MIN_CHAR_COUNT = 5


def strip_specials(text: str) -> str:
    return SPECIAL_STRIP_RE.sub("", text)


def split_pieces(text: str) -> list[str]:
    out: list[str] = []
    for part in SPECIAL_RE.split(text):
        if not part:
            continue
        if part in SPECIAL_SET:
            out.append(part)
        else:
            out.extend(PIECE_RE.findall(part))
    return out


class Tokenizer:
    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = token_to_id or {}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def train(self, texts: list[str], vocab_size: int) -> None:
        char_counts: Counter[str] = Counter()
        piece_counts: Counter[str] = Counter()
        for text in texts:
            clean = strip_specials(text)
            char_counts.update(clean)
            for part in SPECIAL_RE.split(text):
                if part and part not in SPECIAL_SET:
                    piece_counts.update(PIECE_RE.findall(part))

        vocab = list(SPECIAL_TOKENS)
        seen = set(vocab)
        base = list(string.printable) + sorted(
            c for c, n in char_counts.items() if n >= MIN_CHAR_COUNT
        )
        for ch in base:
            if ch not in seen:
                vocab.append(ch)
                seen.add(ch)

        if len(vocab) > vocab_size:
            raise ValueError(
                f"vocab_size {vocab_size} is smaller than the {len(vocab)} "
                f"special + character tokens required"
            )

        for piece, _ in piece_counts.most_common():
            if len(vocab) >= vocab_size:
                break
            if piece in seen:
                continue
            vocab.append(piece)
            seen.add(piece)

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id["<unk>"]
        t2i = self.token_to_id
        ids: list[int] = []
        for piece in split_pieces(text):
            tid = t2i.get(piece)
            if tid is not None:
                ids.append(tid)
            else:
                for ch in piece:
                    ids.append(t2i.get(ch, unk))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, "") for i in ids]
        return "".join(tokens).strip()

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for tok, i in self.token_to_id.items():
            h.update(f"{i}\t{tok}\n".encode("utf-8"))
        return h.hexdigest()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        token_to_id = json.loads(path.read_text(encoding="utf-8"))
        return cls(token_to_id)
