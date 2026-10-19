from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ml_llm.data.tokenizer import Tokenizer


def load_conversations(dataset_dir: Path) -> list[str]:
    formatted = []
    for path in sorted(dataset_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            turns = row["turns"] if "turns" in row else row
            text = ""
            for turn in turns:
                role = "<user>" if turn["role"] == "user" else "<assistant>"
                text += f"{role} {turn['content']} "
            formatted.append(text.strip())
    return formatted


class ConversationDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Tokenizer, context: int):
        self.context = context
        self.examples: list[list[int]] = []
        bos, eos = tokenizer.token_to_id["<bos>"], tokenizer.token_to_id["<eos>"]
        for text in texts:
            ids = [bos] + tokenizer.encode(text) + [eos]
            for i in range(0, max(1, len(ids) - context), context):
                chunk = ids[i:i + context + 1]
                if len(chunk) >= 2:
                    self.examples.append(chunk)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.examples[idx]
        x = chunk[:-1]
        y = chunk[1:]
        pad_len = self.context - len(x)
        if pad_len > 0:
            x = x + [0] * pad_len
            y = y + [-1] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
