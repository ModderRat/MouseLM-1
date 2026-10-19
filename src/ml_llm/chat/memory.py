from __future__ import annotations

import json
import time
from pathlib import Path


class ChatMemory:
    def __init__(self, path: Path, max_turns: int = 20):
        self.path = path
        self.max_turns = max_turns
        self.turns: list[dict] = []
        if path.exists():
            self.turns = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content, "ts": time.time()})
        self.turns = self.turns[-self.max_turns:]
        with self.path.open("a") as f:
            f.write(json.dumps({"role": role, "content": content, "ts": time.time()}) + "\n")

    def recent(self, n: int) -> list[dict]:
        return self.turns[-n:]

    def clear(self) -> None:
        self.turns = []
        self.path.write_text("")
