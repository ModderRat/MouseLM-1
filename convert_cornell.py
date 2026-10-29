from datasets import load_dataset
import json

ds = load_dataset("mylesmharrison/cornell-movie-dialog", split="train")

out_path = "cornell_converted.jsonl"
count = 0
lines = []
for row in ds:
    text = row.get("text") or row.get("line") or list(row.values())[0]
    if text and text.strip():
        lines.append(text.strip())

with open(out_path, "w") as f:
    for i in range(0, len(lines) - 1, 2):
        turns = [
            {"role": "user", "content": lines[i]},
            {"role": "assistant", "content": lines[i + 1]},
        ]
        f.write(json.dumps({"turns": turns}) + "\n")
        count += 1

print(f"Wrote {count} conversations to {out_path}")
