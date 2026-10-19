from __future__ import annotations

import argparse
import platform
import shutil
import sys
import zipfile
from pathlib import Path

import torch
from rich.console import Console

from ml_llm.chat.generate import reply
from ml_llm.chat.memory import ChatMemory
from ml_llm.config import Config
from ml_llm.train.loop import (
    DEFAULT_HF_REPO_ID,
    build_tokenizer,
    latest_checkpoint,
    load_model,
    train,
    upload_to_huggingface,
)

console = Console()

DEBUG_FILES = (
    "config.json",
    "pyproject.toml",
    "diag.py",
    "examples_cache.bin.meta.json",
    "tokenizer.json",
    "old/tokenizer.json",
)


def cmd_train(cfg: Config, args: argparse.Namespace) -> None:
    train(cfg)


def cmd_chat(cfg: Config, args: argparse.Namespace) -> None:
    cfg.paths.ensure()
    tokenizer = build_tokenizer(cfg)
    model, step, _ = load_model(cfg, tokenizer)
    if step == 0:
        console.print("[red]No trained checkpoint found. Run 'ml-llm train' first.[/red]")
        return
    model.eval()
    device = torch.device(cfg.train.device)
    model.to(device)
    memory = ChatMemory(cfg.paths.memory)

    console.print(f"[green]Loaded checkpoint at step {step}. Type 'exit' to quit.[/green]")
    while True:
        try:
            user_input = console.input("[bold cyan]you>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        text = reply(
            model,
            tokenizer,
            memory,
            user_input,
            device,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        console.print(f"[bold magenta]assistant>[/bold magenta] {text}")


def cmd_status(cfg: Config, args: argparse.Namespace) -> None:
    ckpt = latest_checkpoint(cfg)
    console.print(f"root: {cfg.paths.root}")
    console.print(f"checkpoint: {ckpt.name if ckpt else 'none'}")
    console.print(f"tokenizer: {'present' if cfg.paths.tokenizer.exists() else 'missing'}")


def cmd_healthcheck(cfg: Config, args: argparse.Namespace) -> None:
    console.print(f"python: {sys.version.split()[0]}")
    console.print(f"platform: {platform.machine()} / {platform.system()}")
    console.print(f"torch: {torch.__version__}")
    console.print(f"threads: {torch.get_num_threads()}")
    ckpt = latest_checkpoint(cfg)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        ok = state.get("finite", True)
        color = "green" if ok else "red"
        console.print(f"[{color}]latest checkpoint finite: {ok}[/{color}]")
    else:
        console.print("[yellow]no checkpoint found[/yellow]")


def cmd_upload(cfg: Config, args: argparse.Namespace) -> None:
    ckpt = latest_checkpoint(cfg)
    if ckpt is None:
        console.print("[red]No checkpoint found to upload.[/red]")
        return
    step = int(ckpt.stem.split("_")[-1])
    repo_id = getattr(cfg.train, "hf_repo_id", DEFAULT_HF_REPO_ID)
    upload_to_huggingface(cfg=cfg, checkpoint_path=ckpt, repo_id=repo_id, step=step)


def _is_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "ml_llm").is_dir()


def _debug_source_root(cfg: Config) -> Path:
    """Find the checkout containing the files needed for a diagnostic bundle."""
    candidates = (
        Path(cfg.paths.root).expanduser().resolve(),
        Path(__file__).resolve().parents[2],
        Path.cwd().resolve(),
    )
    for candidate in candidates:
        if _is_repo_root(candidate):
            return candidate
    raise RuntimeError(
        "Could not find the ml-llm source checkout. Run this command from the "
        "project directory, or ensure ~/ml_llm_pi contains pyproject.toml and src/ml_llm/."
    )


def _ignore_generated(_directory: str, names: list[str]) -> set[str]:
    """Keep temporary Python/macOS files out of diagnostic bundles."""
    ignored: set[str] = set()
    for name in names:
        if name == "__pycache__" or name == ".DS_Store" or name.endswith(".pyc"):
            ignored.add(name)
    return ignored


def cmd_debugzip(cfg: Config, args: argparse.Namespace) -> None:
    """Create a small source/config ZIP without large datasets or checkpoints."""
    try:
        root = _debug_source_root(cfg)
        desktop = Path.home() / "Desktop"
        desktop.mkdir(parents=True, exist_ok=True)
        out = desktop / "ml_llm_debug_bundle"
        archive = desktop / "ml_llm_debug_bundle.zip"

        # Refresh only the known output folder created by this command.
        if out.exists():
            if out.is_dir():
                shutil.rmtree(out)
            else:
                out.unlink()
        out.mkdir(parents=True)

        copied: list[str] = []
        for name in DEBUG_FILES:
            src = root / name
            if not src.is_file():
                continue
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(name)

        package_source = root / "src" / "ml_llm"
        if package_source.is_dir():
            package_dest = out / "src" / "ml_llm"
            shutil.copytree(
                package_source,
                package_dest,
                ignore=_ignore_generated,
            )
            copied.extend(
                str(path.relative_to(out))
                for path in package_dest.rglob("*")
                if path.is_file()
            )

        if not copied:
            raise RuntimeError(f"No diagnostic files found in {root}")

        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(out))

        console.print(f"[green]Diagnostic bundle created:[/green] {archive}")
        console.print("Included files:")
        for name in sorted(set(copied)):
            console.print(f"  {name}")
        console.print(
            "[dim]Datasets, training checkpoints, chat memory, and logs are not included.[/dim]"
        )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        console.print(f"[red]Could not create diagnostic bundle:[/red] {exc}")
        raise SystemExit(1) from exc


def main() -> None:
    parser = argparse.ArgumentParser(prog="ml-llm")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "status", "healthcheck", "upload", "debugzip"):
        sub.add_parser(name)

    chat_parser = sub.add_parser("chat")
    chat_parser.add_argument("--temperature", type=float, default=0.6)
    chat_parser.add_argument("--top-k", type=int, default=40)
    chat_parser.add_argument("--repetition-penalty", type=float, default=1.3)

    args = parser.parse_args()
    cfg = Config.default()

    handlers = {
        "train": cmd_train,
        "chat": cmd_chat,
        "status": cmd_status,
        "healthcheck": cmd_healthcheck,
        "upload": cmd_upload,
        "debugzip": cmd_debugzip,
    }
    handlers[args.command](cfg, args)


if __name__ == "__main__":
    main()
