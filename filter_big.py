import json, os
root = os.path.expanduser("~/ml_llm_pi")
cap = 300000
for name in ["sharegpt", "wildchat", "lamini"]:
    src = f"{root}/dataset/{name}_converted.jsonl"
    dst = f"{root}/dataset_big/{name}_converted.jsonl"
    os.rename(src, dst)
    kept = 0
    with open(dst) as fin, open(f"{root}/dataset/{name}_filtered.jsonl", "w") as fout:
        for line in fin:
            try:
                turns = json.loads(line)["turns"]
            except Exception:
                continue
            if not turns or len(turns) > 6:
                continue
            ok = True
            for t in turns:
                c = t["content"]
                if len(c) > 500 or len(c) < 2 or "```" in c or "http" in c or "\n\n\n" in c:
                    ok = False
                    break
                if sum(ch.isascii() for ch in c) / len(c) < 0.97:
                    ok = False
                    break
            if ok:
                fout.write(line)
                kept += 1
                if kept >= cap:
                    break
    print(name, kept)
