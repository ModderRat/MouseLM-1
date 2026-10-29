import json
import numpy as np
import torch
from ml_llm.config import ModelConfig
from ml_llm.model.architecture import TinyGPT

mc = ModelConfig(**json.load(open("config.json")))
mc.dropout = 0.0
dev = torch.device("mps")
m = TinyGPT(mc).to(dev)
m.train()
meta = json.load(open("examples_cache.bin.meta.json"))
d = np.memmap("examples_cache.bin", dtype=meta["dtype"], mode="r", shape=(meta["num_rows"], meta["context_len"]))
row = d[5].astype(np.int64)
x = torch.tensor(row[:-1])[None].to(dev)
y = torch.tensor(row[1:])[None].clone().to(dev)
y[y == 0] = -1
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
for i in range(301):
    _, loss = m(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if i % 25 == 0:
        print(i, round(loss.item(), 4), flush=True)
