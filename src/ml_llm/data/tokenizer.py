from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]
TOKEN_RE = re.compile(r"<\w+>|\s+|\w+|[^\w\s]")


class Tokenizer:
    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = token_to_id or {}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def train(self, texts: list[str], vocab_size: int) -> None:
        chars = sorted({c for text in texts for c in text})
        vocab = list(SPECIAL_TOKENS) + chars

        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(TOKEN_RE.findall(text))

        remaining = vocab_size - len(vocab)
        for word, _ in word_counts.most_common():
            if remaining <= 0:
                break
            if word in vocab:
                continue
            vocab.append(word)
            remaining -= 1

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id["<unk>"]
        ids = []
        for piece in TOKEN_RE.findall(text):
            if piece in self.token_to_id:
                ids.append(self.token_to_id[piece])
            else:
                for ch in piece:
                    ids.append(self.token_to_id.get(ch, unk))
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token.get(i, "") for i in ids]
        return "".join(tokens).strip()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id))

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        token_to_id = json.loads(path.read_text())
        return cls(token_to_id)
