from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("ML_LLM_ROOT", str(Path.home() / "ml_llm_pi"))).expanduser()


@dataclass
class ModelConfig:
    dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
    context: int = 1024
    dropout: float = 0.1
    vocab_size: int = 8192
    n_loops: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 100000
    grad_clip: float = 1.0
    eval_every: int = 500
    checkpoint_every: int = 500
    num_workers: int = 0
    device: str = "mps"
    dtype: str = "float32"


@dataclass
class Paths:
    root: Path = DEFAULT_ROOT
    checkpoints: Path = DEFAULT_ROOT / "checkpoints"
    tokenizer: Path = DEFAULT_ROOT / "tokenizer.json"
    dataset: Path = DEFAULT_ROOT / "dataset"
    memory: Path = DEFAULT_ROOT / "memory.jsonl"
    logs: Path = DEFAULT_ROOT / "logs"

    def ensure(self) -> None:
        for p in (self.root, self.checkpoints, self.dataset, self.logs):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    paths: Paths

    @classmethod
    def default(cls) -> "Config":
        paths = Paths()
        model = ModelConfig()
        overlay = paths.root / "config.json"
        if overlay.is_file():
            try:
                data = json.loads(overlay.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            data = data.get("model", data) if isinstance(data, dict) else {}
            for f in fields(ModelConfig):
                if f.name in data:
                    setattr(model, f.name, data[f.name])
        return cls(model=model, train=TrainConfig(), paths=paths)

    @classmethod
    def load(cls, path: Path) -> "Config":
        data = json.loads(path.read_text())

        return cls(
            model=ModelConfig(**data["model"]),
            train=TrainConfig(**data["train"]),
            paths=Paths(
                **{
                    k: Path(v)
                    for k, v in data["paths"].items()
                }
            ),
        )

    def save(self, path: Path) -> None:
        payload = {
            "model": asdict(self.model),
            "train": asdict(self.train),
            "paths": {
                k: str(v)
                for k, v in asdict(self.paths).items()
            },
        }

        path.write_text(
            json.dumps(payload, indent=2)
        )
