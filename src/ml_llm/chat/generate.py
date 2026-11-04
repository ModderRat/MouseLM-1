import torch
import torch.nn.functional as F
from collections import Counter

SPECIAL = ["<bos>", "<eos>", "<user>", "<assistant>", "<pad>", "<unk>"]


def clean_generated_text(text: str) -> str:
    for token in SPECIAL:
        text = text.replace(token, "")
    return " ".join(text.split())


def collapse_repeats(text: str) -> str:
    out = []
    for w in text.split():
        if out and w.lower() == out[-1].lower():
            continue
        out.append(w)
    return " ".join(out)


def trim_to_sentence(text: str) -> str:
    cut = max(text.rfind("."), text.rfind("?"), text.rfind("!"))
    if cut >= 8:
        return text[: cut + 1]
    return text


def is_degenerate(text: str) -> bool:
    words = text.lower().split()
    if len(words) < 4:
        return False
    return len(set(words)) / len(words) < 0.55


def banned_ngram(generated, n):
    if n < 2 or len(generated) < n - 1:
        return set()
    prefix = tuple(generated[-(n - 1):])
    banned = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
    return banned


def build_prompt(memory, user_input: str, max_turns: int = 2) -> str:
    parts = []
    if max_turns > 0:
        for turn in memory.recent(max_turns):
            role = "<user>" if turn["role"] == "user" else "<assistant>"
            parts.append(f"{role} {turn['content'].strip()}")
    parts.append(f"<user> {user_input.strip()}")
    parts.append("<assistant>")
    return "<bos> " + " ".join(parts)


def reply(model, tokenizer, memory, user_input, device, max_new_tokens=60, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.3, history_turns=2, no_repeat_ngram=3):
    if not user_input.strip():
        return ""
    model.eval()
    cfg = getattr(model, "cfg", None) or getattr(getattr(model, "_orig_mod", None), "cfg", None)
    context = getattr(cfg, "context", 1024)
    prompt_ids = tokenizer.encode(build_prompt(memory, user_input, history_turns))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    t2i = tokenizer.token_to_id
    stop_ids = {t2i[t] for t in ("<eos>", "<user>") if t in t2i}
    block = {t2i[t] for t in ("<unk>", "<pad>", "<bos>", "<assistant>") if t in t2i}
    block_ids = sorted(block)
    penalty_ids = {i for t, i in t2i.items() if len(t) > 1 and any(c.isalpha() for c in t) and t not in SPECIAL}
    generated = []
    stopped = False

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids[:, -context:])
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :].float() / max(temperature, 1e-5)

            logits[:, block_ids] = -float("inf")

            if repetition_penalty != 1.0 and generated:
                counts = Counter(t for t in generated if t in penalty_ids)
                for tid, c in counts.items():
                    factor = repetition_penalty ** min(c, 3)
                    v = logits[0, tid]
                    logits[0, tid] = v * factor if v < 0 else v / factor

            if generated and generated[-1] in penalty_ids:
                logits[0, generated[-1]] = -float("inf")

            bans = banned_ngram(generated, no_repeat_ngram)
            if bans:
                logits[0, sorted(bans)] = -float("inf")

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
                stopped = True
                break
            generated.append(tid)
            input_ids = torch.cat((input_ids, next_token), dim=1)

    text = collapse_repeats(clean_generated_text(tokenizer.decode(generated)))
    if not stopped:
        text = trim_to_sentence(text)
    if text and not is_degenerate(text):
        memory.add("user", user_input)
        memory.add("assistant", text)
    return text
