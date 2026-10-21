import json, glob, os
import numpy as np
import torch
from ml_llm.config import ModelConfig
from ml_llm.model.architecture import TinyGPT

mc = ModelConfig(**json.load(open("config.json")))
f = max(glob.glob("checkpoints/step_*.pt"), key=os.path.getmtime)
c = torch.load(f, map_location="cpu", weights_only=False)
m = TinyGPT(mc)
m.load_state_dict(c["model"])
m.eval()
meta = json.load(open("examples_cache.bin.meta.json"))
d = np.memmap("examples_cache.bin", dtype=meta["dtype"], mode="r", shape=(meta["num_rows"], meta["context_len"]))
t = json.load(open("tokenizer.json"))
sp_id = t.get(" ", -1)
inv = {v: k for k, v in t.items()}
n = meta["num_rows"]
print(f, "vocab", len(t), "max_id_in_cache", int(d[:2000].max()))
for i in [0, n // 3, 2 * n // 3, n - 1]:
    row = d[i].astype(np.int64)
    x = torch.tensor(row[:-1])[None]
    y = torch.tensor(row[1:])[None].clone()
    y[y == 0] = -1
    with torch.no_grad():
        _, l = m(x, y)
    print(i, "loss", round(l.item(), 3), "space_frac", round(float((row == sp_id).mean()), 3))
    print("".join(inv.get(int(j), "?") for j in row[:80]))
