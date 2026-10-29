import json
import os
import shutil
from pathlib import Path

root = Path(os.environ.get("ML_LLM_ROOT", str(Path.home() / "ml_llm_pi")))
dataset = root / "dataset"
big = root / "dataset_big"
big.mkdir(parents=True, exist_ok=True)

caps = {"sharegpt": 300000, "wildchat": 300000, "lamini": None}


def keep(turns):
    if not turns or len(turns) > 6:
        return False
    for t in turns:
        c = t.get("content") or ""
        if len(c) > 500 or len(c) < 2 or "```" in c or "http" in c or "\n\n\n" in c:
            return False
        if sum(ch.isascii() for ch in c) / len(c) < 0.97:
            return False
    return True


for name, cap in caps.items():
    src = dataset / f"{name}_converted.jsonl"
    dst = big / f"{name}_converted.jsonl"
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
    if not dst.exists():
        print(name, "missing, skipped")
        continue
    kept = 0
    with dst.open(encoding="utf-8") as fin, (dataset / f"{name}_filtered.jsonl").open("w", encoding="utf-8") as fout:
        for line in fin:
            try:
                turns = json.loads(line)["turns"]
            except Exception:
                continue
            if not keep(turns):
                continue
            fout.write(line)
            kept += 1
            if cap and kept >= cap:
                break
    print(name, kept)
