from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_ROOT = Path.home() / "ml_llm_pi"


@dataclass
class ModelConfig:
    dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
    context: int = 1024
    dropout: float = 0.1
    vocab_size: int = 16384
    n_loops: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 300000
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
        return cls(
            model=ModelConfig(),
            train=TrainConfig(),
            paths=Paths(),
        )

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
