from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


def from_chatterbot_corpus() -> list[list[dict]]:
    import chatterbot_corpus

    base = Path(chatterbot_corpus.__file__).parent / "data" / "english"
    out = []
    for path in sorted(base.glob("*.yml")):
        doc = yaml.safe_load(path.read_text())
        for convo in doc.get("conversations", []):
            turns = []
            for i, line in enumerate(convo):
                role = "user" if i % 2 == 0 else "assistant"
                turns.append({"role": role, "content": line})
            if len(turns) >= 2:
                out.append(turns)
    return out


def main() -> None:
    out_path = Path(os.environ.get("ML_LLM_ROOT", str(Path.home() / "ml_llm_pi"))) / "dataset" / "corpora.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = from_chatterbot_corpus()
    with out_path.open("w") as f:
        for turns in data:
            f.write(json.dumps({"turns": turns}) + "\n")

    print(f"wrote {len(data)} conversations to {out_path}")


if __name__ == "__main__":
    main()
