import json, random, torch
from ml_llm.config import Config
from ml_llm.data.streaming import _format_turns
from ml_llm.train.loop import build_tokenizer, load_model

cfg = Config.default()
tok = build_tokenizer(cfg)
model, _, _ = load_model(cfg, tok)
model.eval()
sp = tok.token_to_id[" "]
rows = [json.loads(l) for l in (cfg.paths.dataset / "dailydialog.jsonl").read_text().splitlines()[:2000] if l.strip()]
random.seed(0)
random.shuffle(rows)
batch = [tok.encode(_format_turns(r["turns"]))[: cfg.model.context + 1] for r in rows[:50]]
for step in (37500, 38000, 38500):
    model.load_state_dict(torch.load(f"checkpoints/step_{step:07d}.pt", map_location="cpu")["model"])
    tot, n = 0.0, 0
    for ids in batch:
        if len(ids) < 8:
            continue
        x = torch.tensor([ids[:-1]])
        y = torch.tensor([ids[1:]])
        with torch.no_grad():
            logits, _ = model(x)
        l = torch.nn.functional.cross_entropy(logits[0], y[0], reduction="none")
        m = y[0] != sp
        tot += l[m].sum().item()
        n += m.sum().item()
    print(step, "word loss:", round(tot / n, 3))
