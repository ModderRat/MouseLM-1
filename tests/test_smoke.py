from pathlib import Path

from ml_llm.config import Config, ModelConfig, Paths, TrainConfig
from ml_llm.data.tokenizer import Tokenizer


def test_package_and_config_modules_import():
    import ml_llm
    from ml_llm import config

    assert ml_llm is not None
    assert config.Config is Config


def test_config_save_and_load_round_trip(tmp_path: Path):
    config = Config(
        model=ModelConfig(dim=64, n_heads=4, n_layers=2, context=32),
        train=TrainConfig(batch_size=2, grad_accum=3, device="cpu"),
        paths=Paths(
            root=tmp_path,
            checkpoints=tmp_path / "checkpoints",
            tokenizer=tmp_path / "tokenizer.json",
            dataset=tmp_path / "dataset",
            memory=tmp_path / "memory.jsonl",
            logs=tmp_path / "logs",
        ),
    )
    config_path = tmp_path / "config.json"

    config.save(config_path)
    loaded = Config.load(config_path)

    assert loaded.model.dim == 64
    assert loaded.model.n_heads == 4
    assert loaded.model.n_layers == 2
    assert loaded.train.batch_size == 2
    assert loaded.train.grad_accum == 3
    assert loaded.train.device == "cpu"
    assert loaded.paths.root == tmp_path


def test_paths_ensure_creates_runtime_directories(tmp_path: Path):
    paths = Paths(
        root=tmp_path,
        checkpoints=tmp_path / "checkpoints",
        tokenizer=tmp_path / "tokenizer.json",
        dataset=tmp_path / "dataset",
        memory=tmp_path / "memory.jsonl",
        logs=tmp_path / "logs",
    )

    paths.ensure()

    assert paths.root.is_dir()
    assert paths.checkpoints.is_dir()
    assert paths.dataset.is_dir()
    assert paths.logs.is_dir()


def test_tokenizer_encode_decode_and_persistence(tmp_path: Path):
    tokenizer = Tokenizer()
    tokenizer.train(["hello world", "hello pip"], vocab_size=256)

    encoded = tokenizer.encode("hello world")
    assert encoded
    assert tokenizer.decode(encoded) == "hello world"

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    loaded = Tokenizer.load(tokenizer_path)

    assert loaded.encode("hello world") == encoded
    assert loaded.fingerprint() == tokenizer.fingerprint()
