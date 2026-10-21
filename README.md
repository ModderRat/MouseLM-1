ml-llm v7 (MouseLM-1)

Install:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .
  pip install -e ".[data]"

Root folder: set ML_LLM_ROOT to the project folder (default ~/ml_llm_pi).
Checkpoints, tokenizer.json, examples_cache.bin, memory.jsonl and logs live there.
Put *.jsonl training files in $ML_LLM_ROOT/dataset.
  format per line: {"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
Optional $ML_LLM_ROOT/config.json overrides model size (dim, n_heads, n_layers, context, dropout, vocab_size, n_loops).

Commands:
  ml-llm train
  ml-llm chat
  ml-llm status
  ml-llm healthcheck

Steps counted in train are optimizer updates (one per grad_accum micro-batches).
Changing the tokenizer invalidates checkpoints: delete tokenizer.json, tokenizer.fingerprint, examples_cache.bin and the checkpoints folder to start fresh.
