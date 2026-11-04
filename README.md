# ml-llm v7 — MouseLM-1

A tiny GPT trainer and chat tool designed for local machines, including Raspberry Pi and Apple Silicon Macs.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e ".[data]"
```

## Runtime files

The project root is controlled by `ML_LLM_ROOT` and defaults to `~/ml_llm_pi`.

Checkpoints, `tokenizer.json`, `examples_cache.bin`, `memory.jsonl`, and logs are stored under that root. Put `*.jsonl` training files in `$ML_LLM_ROOT/dataset`.

Each JSONL line should contain conversation turns:

```json
{"turns": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

An optional `config.json` can override model settings such as `dim`, `n_heads`, `n_layers`, `context`, `dropout`, `vocab_size`, and `n_loops`.

## Commands

```bash
ml-llm train
ml-llm chat
ml-llm status
ml-llm healthcheck
ml-llm upload
ml-llm debugzip
```

## Training notes

Steps counted by the trainer are optimizer updates (one per `grad_accum` micro-batches).

The default model configuration is defined in `src/ml_llm/config.py`. Increase context length or model dimensions only after checking available memory.

Changing the tokenizer invalidates existing checkpoints. To start fresh after changing it, remove `tokenizer.json`, `tokenizer.fingerprint`, `examples_cache.bin`, and the checkpoints folder.
