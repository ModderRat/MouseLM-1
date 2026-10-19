from datasets import load_dataset
import json

ds = load_dataset("ajibawa-2023/Code-290k-ShareGPT", split="train")

out_path = "code_converted.jsonl"
count = 0

with open(out_path, "w") as f:
    for row in ds:
        convo = row.get("conversations") or []
        turns = []
        for msg in convo:
            role_raw = (msg.get("from") or "").lower()
            content = msg.get("value") or ""
            if not content.strip():
                continue
            role = "user" if role_raw in ("human", "user") else "assistant"
            turns.append({"role": role, "content": content.strip()})
        if len(turns) >= 2:
            f.write(json.dumps({"turns": turns}) + "\n")
            count += 1

print(f"Wrote {count} conversations to {out_path}")
