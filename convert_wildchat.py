import json
import re
import html
import urllib.request

TAG_RE = re.compile(r"<[^>]+>")
STRAY_RE = re.compile(r"[<>]")


def clean(text: str) -> str:
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = STRAY_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


out_path = "wildchat_converted.jsonl"
count = 0
base_url = "https://huggingface.co/datasets/bigstupidhats/wildchat-en/resolve/main/wildchat-{}-of-10.jsonl"
with open(out_path, "w") as out_f:
    for shard in range(10):
        url = base_url.format(shard)
        print(f"Fetching shard {shard}...")
        with urllib.request.urlopen(url) as resp:
            for line in resp:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                convo = row.get("conversation") or []
                turns = []
                for msg in convo:
                    role_raw = (msg.get("role") or "").lower()
                    content = msg.get("content") or ""
                    content = clean(content)
                    if not content:
                        continue
                    role = "user" if role_raw == "user" else "assistant"
                    turns.append({"role": role, "content": content})
                if len(turns) >= 2:
                    out_f.write(json.dumps({"turns": turns}) + "\n")
                    count += 1
print(f"Wrote {count} conversations to {out_path}")
