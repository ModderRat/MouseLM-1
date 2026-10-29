from datasets import load_dataset
import json

ds = load_dataset("awsaf49/persona-chat", split="train")

out_path = "personachat_converted.jsonl"
count = 0
conv_turns = []

def flush(turns, f):
    global count
    if len(turns) >= 2:
        f.write(json.dumps({"turns": turns}) + "\n")
        count += 1

with open(out_path, "w") as f:
    for row in ds:
        line = row["text"]
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        idx_str, rest = parts
        try:
            idx = int(idx_str)
        except ValueError:
            continue

        if idx == 1 and conv_turns:
            flush(conv_turns, f)
            conv_turns = []

        if "your persona:" in rest or "partner's persona:" in rest:
            continue

        if "\t" in rest:
            fields = rest.split("\t")
            user_utt = fields[0].strip()
            bot_utt = fields[1].strip() if len(fields) > 1 else ""
            if user_utt:
                conv_turns.append({"role": "user", "content": user_utt})
            if bot_utt:
                conv_turns.append({"role": "assistant", "content": bot_utt})

    flush(conv_turns, f)

print(f"Wrote {count} conversations to {out_path}")
