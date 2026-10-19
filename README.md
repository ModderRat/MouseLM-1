ml-llm v7 — macOS build

Install:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .

Data: put *.jsonl files in ~/.ml_llm/dataset
  format per line: {"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Commands:
  ml-llm train
  ml-llm chat
  ml-llm status
  ml-llm healthcheck

Config defaults (src/ml_llm/config.py): 192 dim, 4 heads, 6 layers, 256 context, fp32, batch 6 x grad_accum 4.
Increase context/dim only after confirming RAM headroom with healthcheck.
