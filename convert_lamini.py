from datasets import load_dataset
import json

ds = load_dataset("MBZUAI/LaMini-instruction", split="train")

out_path = "lamini_converted.jsonl"
count = 0

with open(out_path, "w") as f:
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("response") or "").strip()
        if not instruction or not response:
            continue
        turns = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        f.write(json.dumps({"turns": turns}) + "\n")
        count += 1

print(f"Wrote {count} conversations to {out_path}")
