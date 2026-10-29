<<<<<<< HEAD
ml-llm v7 (MouseLM-1)
=======
ml-llm v7 — macOS build
>>>>>>> ec97c23a749936183d772472a77460210dc863e3

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

<<<<<<< HEAD
Steps counted in train are optimizer updates (one per grad_accum micro-batches).
Changing the tokenizer invalidates checkpoints: delete tokenizer.json, tokenizer.fingerprint, examples_cache.bin and the checkpoints folder to start fresh.
=======
Config defaults (src/ml_llm/config.py): 192 dim, 4 heads, 6 layers, 256 context, fp32, batch 6 x grad_accum 4.
Increase context/dim only after confirming RAM headroom with healthcheck.
>>>>>>> ec97c23a749936183d772472a77460210dc863e3
