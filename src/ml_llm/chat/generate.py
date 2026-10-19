import torch
import torch.nn.functional as F

SPECIAL = ["<bos>", "<eos>", "<user>", "<assistant>", "<pad>", "<unk>"]


def clean_generated_text(text: str) -> str:
    for token in SPECIAL:
        text = text.replace(token, "")
    return " ".join(text.split())


def build_prompt(memory, user_input: str, max_turns: int = 2) -> str:
    parts = []
    if max_turns > 0:
        for turn in memory.recent(max_turns):
            role = "<user>" if turn["role"] == "user" else "<assistant>"
            parts.append(f"{role} {turn['content'].strip()}")
    parts.append(f"<user> {user_input.strip()}")
    parts.append("<assistant>")
    return "<bos> " + " ".join(parts)


def reply(model, tokenizer, memory, user_input, device, max_new_tokens=150, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.15, history_turns=2):
    model.eval()
    context = getattr(getattr(model, "config", None), "context", 1024)
    prompt_ids = tokenizer.encode(build_prompt(memory, user_input, history_turns))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    t2i = tokenizer.token_to_id
    stop_ids = {t2i[t] for t in ("<eos>", "<user>") if t in t2i}
    block = {t2i[t] for t in ("<unk>", "<pad>", "<bos>", "<assistant>") if t in t2i}
    block |= {i for t, i in t2i.items() if len(t) == 1 and t.isalnum() and t not in "aAI"}
    block_ids = sorted(block)
    penalty_ids = {i for t, i in t2i.items() if len(t) > 1 and any(c.isalpha() for c in t) and t not in SPECIAL}
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids[:, -context:])
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)

            logits[:, block_ids] = -float("inf")

            pen = sorted(set(generated) & penalty_ids)
            if repetition_penalty != 1.0 and pen:
                idx = torch.tensor(pen, device=logits.device)
                vals = logits[0, idx]
                logits[0, idx] = torch.where(vals < 0, vals * repetition_penalty, vals / repetition_penalty)

            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits[0, sorted_idx[0][remove[0]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tid = next_token.item()
            if tid in stop_ids:
                break
            generated.append(tid)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    text = clean_generated_text(tokenizer.decode(generated))
    memory.add("user", user_input)
    memory.add("assistant", text)
    return text
