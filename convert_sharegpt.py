from datasets import load_dataset
import json
import re
import html

TAG_RE = re.compile(r"<[^>]+>")
STRAY_RE = re.compile(r"[<>]")


def clean(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = STRAY_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


ds = load_dataset("RyokoAI/ShareGPT52K", split="train")
out_path = "sharegpt_converted.jsonl"
count = 0
with open(out_path, "w") as f:
    for row in ds:
        convo = row.get("conversations") or row.get("items") or []
        turns = []
        for msg in convo:
            role_raw = (msg.get("from") or msg.get("role") or "").lower()
            content = msg.get("value") or msg.get("content") or ""
            content = clean(content)
            if not content:
                continue
            role = "user" if role_raw in ("human", "user") else "assistant"
            turns.append({"role": role, "content": content})
        if len(turns) >= 2:
            f.write(json.dumps({"turns": turns}) + "\n")
            count += 1
print(f"Wrote {count} conversations to {out_path}")
