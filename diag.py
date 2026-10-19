import json
import random

import torch

from ml_llm.config import Config
from ml_llm.data.streaming import _format_turns
from ml_llm.train.loop import build_tokenizer, load_model

cfg = Config.default()
tok = build_tokenizer(cfg)
model, step, _ = load_model(cfg, tok)
model.eval()

path = cfg.paths.dataset / "dailydialog.jsonl"
rows = [json.loads(l) for l in path.read_text().splitlines()[:2000] if l.strip()]
random.seed(0)
random.shuffle(rows)

losses = []
for row in rows[:50]:
    text = _format_turns(row["turns"])
    ids = tok.encode(text)[: cfg.model.context + 1]
    if len(ids) < 8:
        continue
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    with torch.no_grad():
        _, loss = model(x, y)
    losses.append(loss.item())
print("teacher-forced loss:", sum(losses) / len(losses), "n=", len(losses))

ids = tok.encode("<bos> <user> hello <assistant> ")
print("prompt tokens:", [tok.id_to_token[i] for i in ids])
with torch.no_grad():
    logits, _ = model(torch.tensor([ids], dtype=torch.long))
probs = torch.softmax(logits[0, -1], dim=-1)
top = torch.topk(probs, 10)
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(repr(tok.id_to_token[i]), round(p, 4))

x = torch.tensor([tok.encode(_format_turns(rows[0]["turns"]))[:60]], dtype=torch.long)
with torch.no_grad():
    logits, _ = model(x)
pred = logits[0].argmax(-1).tolist()
print("truth:", [tok.id_to_token[i] for i in x[0, 1:].tolist()][:40])
print("pred :", [tok.id_to_token[i] for i in pred][:40])
