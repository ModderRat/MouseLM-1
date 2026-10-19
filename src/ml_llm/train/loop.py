from __future__ import annotations
import sys

import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, random_split

from ml_llm.config import Config
from ml_llm.data.dataset import ConversationDataset, load_conversations
from ml_llm.data.streaming import build_examples_cache, sample_texts, MemmapConversationDataset
from ml_llm.data.tokenizer import Tokenizer
from ml_llm.model.architecture import TinyGPT
from ml_llm.train.distributed import cleanup_distributed, init_distributed, is_distributed, wrap_model


console = Console()

AUTOTUNE_FILENAME = ".autotune.json"
DEFAULT_HF_REPO_ID = "Mouse-AI/MouSLM1"


# ============================================================
# CPU / DEVICE
# ============================================================

def configure_cpu(threads: int | None = None, interop: int | None = None) -> None:
    cpu_count = os.cpu_count() or 4

    if threads is None:
        threads = int(os.environ.get("ML_LLM_THREADS", cpu_count))

    if interop is None:
        interop = int(
            os.environ.get(
                "ML_LLM_INTEROP_THREADS",
                max(1, min(4, cpu_count // 2 or 1)),
            )
        )

    threads = max(1, min(int(threads), cpu_count))
    interop = max(1, min(int(interop), cpu_count))

    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass

    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass

    console.print(
        f"[cyan]CPU threads: {torch.get_num_threads()} | "
        f"interop: {torch.get_num_interop_threads()}[/cyan]"
    )


def resolve_device(cfg: Config) -> torch.device:
    configured = str(
        getattr(cfg.train, "device", "auto")
    ).lower().strip()

    if configured not in ("", "auto"):
        try:
            requested = torch.device(configured)

            if requested.type == "cuda":
                if torch.cuda.is_available():
                    return requested

                console.print(
                    "[yellow]CUDA requested but unavailable; using CPU.[/yellow]"
                )
                return torch.device("cpu")

            if requested.type == "mps":
                if (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    return requested

                console.print(
                    "[yellow]MPS requested but unavailable; using CPU.[/yellow]"
                )
                return torch.device("cpu")

            return requested

        except Exception:
            console.print(
                f"[yellow]Invalid device '{configured}'; using auto.[/yellow]"
            )

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def configure_backend(device: torch.device) -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


# ============================================================
# OPTIONAL COMPILE
# ============================================================

def maybe_compile(model: TinyGPT, device: torch.device) -> TinyGPT:
    if os.environ.get("ML_LLM_COMPILE", "0") != "1":
        return model

    if not hasattr(torch, "compile"):
        console.print(
            "[yellow]torch.compile is unavailable.[/yellow]"
        )
        return model

    console.print(
        "[cyan]Compiling model with torch.compile...[/cyan]"
    )

    try:
        compiled = torch.compile(
            model,
            mode="reduce-overhead",
        )

        console.print(
            "[green]torch.compile enabled.[/green]"
        )

        return compiled

    except Exception as exc:
        console.print(
            f"[yellow]torch.compile failed: {exc}[/yellow]"
        )
        return model


# ============================================================
# AUTOTUNE
# ============================================================

def autotune_path(cfg: Config) -> Path:
    return cfg.paths.checkpoints.parent / AUTOTUNE_FILENAME


def _benchmark_one(
    model: TinyGPT,
    dataset,
    device: torch.device,
    threads: int,
    batch_size: int,
    repeats: int = 3,
) -> float:
    configure_cpu(threads)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    try:
        x, y = next(iter(loader))
    except StopIteration:
        return 0.0

    x = x.to(device)
    y = y.to(device)

    model.train()

    try:
        for _ in range(2):
            model.zero_grad(set_to_none=True)

            _, loss = model(x, y)

            if not torch.isfinite(loss):
                model.zero_grad(set_to_none=True)
                return 0.0

            loss.backward()
            model.zero_grad(set_to_none=True)

    except (RuntimeError, MemoryError):
        model.zero_grad(set_to_none=True)
        return 0.0

    timings = []

    for _ in range(repeats):
        model.zero_grad(set_to_none=True)

        start = time.perf_counter()

        try:
            _, loss = model(x, y)

            if not torch.isfinite(loss):
                model.zero_grad(set_to_none=True)
                return 0.0

            loss.backward()

        except (RuntimeError, MemoryError):
            model.zero_grad(set_to_none=True)
            return 0.0

        elapsed = max(
            time.perf_counter() - start,
            0.0001,
        )

        model.zero_grad(set_to_none=True)

        timings.append(
            int(x.numel()) / elapsed
        )

    if not timings:
        return 0.0

    timings.sort()

    return timings[len(timings) // 2]


def big_autotune(
    cfg: Config,
    model: TinyGPT,
    dataset,
    device: torch.device,
) -> tuple[int, int]:

    if device.type != "cpu":
        return (
            torch.get_num_threads(),
            int(cfg.train.batch_size),
        )

    cache_path = autotune_path(cfg)

    cpu_count = os.cpu_count() or 4

    force = (
        os.environ.get(
            "ML_LLM_REBENCH",
            "0",
        )
        == "1"
    )

    if cache_path.exists() and not force:
        try:
            cached = json.loads(
                cache_path.read_text()
            )

            if int(cached.get("cpu_count", -1)) == cpu_count:
                threads = int(cached["threads"])
                batch = int(cached["batch"])

                configure_cpu(threads)

                cfg.train.batch_size = batch

                console.print(
                    f"[bold green]"
                    f"Saved Pi tuning: "
                    f"{threads} threads / "
                    f"batch {batch}"
                    f"[/bold green]"
                )

                if "tok_s" in cached:
                    console.print(
                        f"[green]"
                        f"Saved benchmark: "
                        f"{float(cached['tok_s']):,.0f} tok/s"
                        f"[/green]"
                    )

                return threads, batch

        except Exception:
            pass

    console.print(
        "[bold cyan]"
        "Running one-time Raspberry Pi performance benchmark..."
        "[/bold cyan]"
    )

    base_batch = max(
        1,
        int(cfg.train.batch_size),
    )

    batch_candidates = sorted(
        {
            1,
            2,
            4,
            8,
            16,
            32,
            base_batch,
            base_batch * 2,
        }
    )

    batch_candidates = [
        b for b in batch_candidates
        if b <= 64
    ]

    thread_candidates = list(
        range(
            1,
            cpu_count + 1,
        )
    )

    if cpu_count > 8:
        thread_candidates = sorted(
            {
                1,
                2,
                4,
                6,
                8,
                cpu_count,
            }
        )

    candidates = [
        (threads, batch)
        for threads in thread_candidates
        for batch in batch_candidates
    ]

    console.print(
        f"[cyan]"
        f"Testing {len(candidates)} configurations..."
        f"[/cyan]"
    )

    best_speed = 0.0
    best_threads = cpu_count
    best_batch = base_batch

    benchmark_start = time.perf_counter()

    for index, (threads, batch) in enumerate(
        candidates,
        start=1,
    ):
        speed = _benchmark_one(
            model=model,
            dataset=dataset,
            device=device,
            threads=threads,
            batch_size=batch,
            repeats=3,
        )

        if speed > 0:
            console.print(
                f"[dim]"
                f"{index}/{len(candidates)} "
                f"{threads} threads / "
                f"batch {batch} = "
                f"{speed:,.0f} tok/s"
                f"[/dim]"
            )
        else:
            console.print(
                f"[yellow]"
                f"{index}/{len(candidates)} "
                f"{threads} threads / "
                f"batch {batch} = failed"
                f"[/yellow]"
            )

        if speed > best_speed:
            best_speed = speed
            best_threads = threads
            best_batch = batch

    configure_cpu(best_threads)

    cfg.train.batch_size = best_batch

    elapsed = max(
        time.perf_counter() - benchmark_start,
        0.001,
    )

    result = {
        "threads": best_threads,
        "batch": best_batch,
        "tok_s": best_speed,
        "cpu_count": cpu_count,
        "benchmark_seconds": elapsed,
        "created": time.time(),
    }

    try:
        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_path.write_text(
            json.dumps(
                result,
                indent=2,
            )
        )

    except Exception as exc:
        console.print(
            f"[yellow]"
            f"Could not save tuning result: {exc}"
            f"[/yellow]"
        )

    console.print(
        "[bold green]"
        "=========================================="
        "[/bold green]"
    )

    console.print(
        f"[bold green]"
        f"BEST: {best_threads} threads / "
        f"batch {best_batch}"
        f"[/bold green]"
    )

    console.print(
        f"[bold green]"
        f"SPEED: {best_speed:,.0f} tokens/sec"
        f"[/bold green]"
    )

    console.print(
        f"[green]"
        f"Saved automatically to {cache_path}"
        f"[/green]"
    )

    console.print(
        "[bold green]"
        "=========================================="
        "[/bold green]"
    )

    return best_threads, best_batch


# ============================================================
# DATASET / TOKENIZER
# ============================================================

def dataset_fingerprint(dataset_dir: Path) -> str:
    h = hashlib.sha256()

    if not dataset_dir.exists():
        return h.hexdigest()

    for path in sorted(
        dataset_dir.glob("*.jsonl")
    ):
        try:
            stat = path.stat()

            h.update(path.name.encode())
            h.update(str(stat.st_size).encode())

        except OSError:
            continue

    return h.hexdigest()


def build_tokenizer(cfg: Config) -> Tokenizer:
    tok_path = cfg.paths.tokenizer
    fp_path = tok_path.with_suffix(".fingerprint")

    current_fp = dataset_fingerprint(
        cfg.paths.dataset
    )

    if (
        tok_path.exists()
        and fp_path.exists()
        and fp_path.read_text().strip()
        == current_fp
    ):
        console.print(
            "[green]Using existing tokenizer.[/green]"
        )
        return Tokenizer.load(tok_path)

    if tok_path.exists():
        console.print(
            "[yellow]"
            "Dataset changed. Rebuilding tokenizer "
            "and clearing incompatible checkpoints."
            "[/yellow]"
        )

        if cfg.paths.checkpoints.exists():
            for ckpt in cfg.paths.checkpoints.glob(
                "step_*.pt"
            ):
                try:
                    ckpt.unlink()
                except OSError:
                    pass

    texts = sample_texts(
        cfg.paths.dataset
    )

    tok = Tokenizer()

    tok.train(
        texts,
        cfg.model.vocab_size,
    )

    tok.save(tok_path)

    fp_path.write_text(
        current_fp
    )

    console.print(
        f"[green]"
        f"Tokenizer ready. "
        f"vocab={tok.vocab_size:,}"
        f"[/green]"
    )

    return tok


# ============================================================
# HUGGING FACE AUTOUPLOAD
# ============================================================

def upload_to_huggingface(
    cfg: Config,
    checkpoint_path: Path,
    repo_id: str = DEFAULT_HF_REPO_ID,
    step: int = 0,
) -> None:

    if not checkpoint_path.exists():
        return

    try:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        from huggingface_hub import HfApi, CommitOperationAdd

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        console.print(
            f"[cyan]Uploading all model parts (step {step}) to Hugging Face ({repo_id})...[/cyan]"
        )

        operations = []

        # 1. Add the main checkpoint file
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"checkpoints/{checkpoint_path.name}",
                path_or_fileobj=str(checkpoint_path)
            )
        )

        # 2. Add the tokenizer files
        if hasattr(cfg, "paths") and hasattr(cfg.paths, "tokenizer"):
            tok_path = cfg.paths.tokenizer
            if tok_path.exists():
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=tok_path.name,
                        path_or_fileobj=str(tok_path)
                    )
                )
            
            fp_path = tok_path.with_suffix(".fingerprint")
            if fp_path.exists():
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=fp_path.name,
                        path_or_fileobj=str(fp_path)
                    )
                )

        # 3. Explicitly export and add model config JSON
        config_path = cfg.paths.root / "config.json"
        
        # Extract model hyperparameter dict from cfg.model
        model_cfg = vars(cfg.model) if hasattr(cfg.model, "__dict__") else dict(cfg.model)
        config_path.write_text(json.dumps(model_cfg, indent=2))

        operations.append(
            CommitOperationAdd(
                path_in_repo="config.json",
                path_or_fileobj=str(config_path)
            )
        )

        # Execute single atomic commit to update all parts at once
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=f"Auto-upload full model (checkpoint + tokenizer + config) step {step:07d}",
        )

        console.print(
            f"[bold green]Successfully uploaded all model parts to Hugging Face![/bold green]"
        )

    except Exception as exc:
        console.print(
            f"[yellow]Hugging Face upload failed for step {step}: {exc}. "
            f"Training will continue normally.[/yellow]"
        )


# ============================================================
# CHECKPOINTS
# ============================================================

def latest_checkpoint(
    cfg: Config,
) -> Path | None:

    if not cfg.paths.checkpoints.exists():
        return None

    ckpts = list(
        cfg.paths.checkpoints.glob(
            "step_*.pt"
        )
    )

    if not ckpts:
        return None

    def step_number(
        path: Path,
    ) -> int:
        try:
            return int(
                path.stem.split("_")[-1]
            )
        except (
            ValueError,
            IndexError,
        ):
            return -1

    ckpts.sort(
        key=step_number
    )

    return ckpts[-1]


def state_is_finite(
    state: Any,
) -> bool:

    if isinstance(
        state,
        dict,
    ):
        return all(
            state_is_finite(value)
            for value in state.values()
        )

    if isinstance(
        state,
        (list, tuple),
    ):
        return all(
            state_is_finite(value)
            for value in state
        )

    if torch.is_tensor(state):
        if (
            state.is_floating_point()
            or state.is_complex()
        ):
            return bool(
                torch.isfinite(
                    state
                ).all().item()
            )

    return True


def load_model(
    cfg: Config,
    tokenizer: Tokenizer,
) -> tuple[
    TinyGPT,
    int,
    dict[str, Any] | None,
]:

    cfg.model.vocab_size = (
        tokenizer.vocab_size
    )

    model = TinyGPT(
        cfg.model
    )

    ckpt = latest_checkpoint(
        cfg
    )

    step = 0
    optimizer_state = None

    if ckpt is None:
        console.print(
            "[cyan]Starting a new model.[/cyan]"
        )

        return model, step, optimizer_state

    console.print(
        f"[cyan]"
        f"Loading checkpoint: {ckpt.name}"
        f"[/cyan]"
    )

    try:
        state = torch.load(
            ckpt,
            map_location="cpu",
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not load checkpoint "
            f"{ckpt}: {exc}"
        ) from exc

    if not state.get(
        "finite",
        True,
    ):
        raise RuntimeError(
            f"{ckpt.name} is marked non-finite."
        )

    model_state = state.get(
        "model"
    )

    if model_state is None:
        raise RuntimeError(
            f"{ckpt.name} does not contain "
            f"a model state."
        )

    if not state_is_finite(
        model_state
    ):
        raise RuntimeError(
            f"{ckpt.name} contains non-finite "
            f"model parameters."
        )

    try:
        model.load_state_dict(
            model_state
        )

    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint {ckpt.name} does not "
            f"match the current model architecture: "
            f"{exc}"
        ) from exc

    step = int(
        state.get(
            "step",
            0,
        )
    )

    optimizer_state = state.get(
        "optimizer"
    )

    console.print(
        f"[green]"
        f"Resumed from {ckpt.name} "
        f"at step {step:,}"
        f"[/green]"
    )

    return (
        model,
        step,
        optimizer_state,
    )


def _prune_old_checkpoints(cfg, current_step: int) -> None:
    ckpt_dir = cfg.paths.checkpoints
    for p in ckpt_dir.glob("step_*.pt"):
        try:
            step_num = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if current_step - step_num > 1000:
            try:
                p.unlink()
            except OSError:
                pass


def save_checkpoint(
    cfg: Config,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    finite: bool,
) -> Path:

    path = (
        cfg.paths.checkpoints
        / f"step_{step:07d}.pt"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_to_save = model

    if hasattr(
        model,
        "_orig_mod",
    ):
        model_to_save = model._orig_mod

    model_state = (
        model_to_save.state_dict()
    )

    if not finite:
        finite = state_is_finite(
            model_state
        )

    payload = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "finite": finite,
        "saved_at": time.time(),
    }

    # Use a unique temporary filename so a crashed/partial save
    # can never be mistaken for a valid checkpoint.
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{int(time.time() * 1000000)}.tmp"
    )

    def cleanup_tmp() -> None:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    def write_checkpoint(
        use_legacy: bool = False,
    ) -> None:
        """
        Write the checkpoint through an explicitly opened file.
        This avoids some filesystem/iostream problems encountered
        when torch.save() is given a Path directly.
        """

        with open(
            tmp_path,
            "wb",
        ) as file:

            torch.save(
                payload,
                file,
                _use_new_zipfile_serialization=not use_legacy,
            )

            file.flush()

            try:
                os.fsync(
                    file.fileno()
                )
            except OSError:
                pass

    def verify_checkpoint() -> None:
        """
        Make sure the temporary file is actually readable before
        replacing the existing checkpoint.
        """

        test_state = torch.load(
            tmp_path,
            map_location="cpu",
        )

        if not isinstance(
            test_state,
            dict,
        ):
            raise RuntimeError(
                "Checkpoint verification failed: "
                "checkpoint is not a dictionary."
            )

        if test_state.get(
            "step"
        ) != step:
            raise RuntimeError(
                "Checkpoint verification failed: "
                f"expected step {step}, "
                f"got {test_state.get('step')}."
            )

        if "model" not in test_state:
            raise RuntimeError(
                "Checkpoint verification failed: "
                "model state is missing."
            )

        if not test_state.get(
            "finite",
            True,
        ):
            raise RuntimeError(
                "Checkpoint verification failed: "
                "checkpoint is marked non-finite."
            )

    try:
        # ----------------------------------------------------
        # FIRST ATTEMPT
        # ----------------------------------------------------

        try:
            write_checkpoint(
                use_legacy=False,
            )

        except Exception as exc:
            console.print(
                f"[yellow]"
                f"Checkpoint ZIP serialization failed at "
                f"step {step:,}: {exc}"
                f"[/yellow]"
            )

            cleanup_tmp()

            # ------------------------------------------------
            # FALLBACK
            # ------------------------------------------------

            console.print(
                "[cyan]"
                "Retrying checkpoint with legacy PyTorch "
                "serialization..."
                "[/cyan]"
            )

            write_checkpoint(
                use_legacy=True,
            )

        # ----------------------------------------------------
        # VERIFY BEFORE REPLACING EXISTING CHECKPOINT
        # ----------------------------------------------------

        verify_checkpoint()

        # ----------------------------------------------------
        # ATOMIC REPLACEMENT
        # ----------------------------------------------------

        tmp_path.replace(
            path
        )

        # ----------------------------------------------------
        # PRUNE ONLY AFTER SUCCESSFUL SAVE
        # ----------------------------------------------------

        _prune_old_checkpoints(
            cfg,
            step,
        )

        console.print(
            f"[green]"
            f"Checkpoint saved: {path.name}"
            f"[/green]"
        )

        return path

    except Exception as exc:

        cleanup_tmp()

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT kill training because a periodic checkpoint
        # failed. The previous checkpoint remains untouched.
        # ----------------------------------------------------

        console.print(
            f"[bold yellow]"
            f"WARNING: checkpoint save failed at "
            f"step {step:,}: {exc}"
            f"[/bold yellow]"
        )

        console.print(
            f"[yellow]"
            f"Previous checkpoint was preserved. "
            f"Training will continue."
            f"[/yellow]"
        )

        # Return the last known-good checkpoint if one exists.
        existing = latest_checkpoint(
            cfg
        )

        if existing is not None:
            return existing

        # If there is no previous checkpoint at all, return the
        # intended path. The caller does not use the returned file
        # until the final upload stage.
        return path
# ============================================================
# DATALOADER
# ============================================================

def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:

    num_workers = max(
        0,
        int(num_workers),
    )

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        kwargs[
            "persistent_workers"
        ] = True

        kwargs[
            "prefetch_factor"
        ] = 2

    return DataLoader(
        dataset,
        **kwargs,
    )


def move_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    if device.type == "cuda":
        return (
            x.to(
                device,
                non_blocking=True,
            ),
            y.to(
                device,
                non_blocking=True,
            ),
        )

    return (
        x.to(device),
        y.to(device),
    )


# ============================================================
# AUTOCAST
# ============================================================

def get_autocast_context(
    device: torch.device,
):

    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    if (
        device.type == "cpu"
        and os.environ.get(
            "ML_LLM_CPU_BF16",
            "0",
        )
        == "1"
    ):
        try:
            return torch.autocast(
                device_type="cpu",
                dtype=torch.bfloat16,
            )
        except Exception:
            pass

    return contextlib.nullcontext()


# ============================================================
# TRAINING
# ============================================================

def train(
    cfg: Config,
) -> None:

    cfg.paths.ensure()

    device = resolve_device(
        cfg
    )

    configure_backend(
        device
    )

    if device.type == "cpu":
        configure_cpu()

    console.print(
        f"[bold green]"
        f"Training device: {device}"
        f"[/bold green]"
    )

    if device.type == "cuda":
        console.print(
            f"[green]"
            f"CUDA GPU: "
            f"{torch.cuda.get_device_name(0)}"
            f"[/green]"
        )

    elif device.type == "mps":
        console.print(
            "[green]"
            "Apple Metal/MPS accelerator enabled."
            "[/green]"
        )

    else:
        console.print(
            "[cyan]"
            "Raspberry Pi CPU training mode."
            "[/cyan]"
        )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = build_tokenizer(
        cfg
    )

    cache_path = (
        cfg.paths.root / "examples_cache.bin"
    )

    num_rows = build_examples_cache(
        cfg.paths.dataset,
        tokenizer,
        cfg.model.context,
        cache_path,
    )

    if num_rows == 0:
        console.print(
            "[red]"
            "No training data found in dataset directory."
            "[/red]"
        )
        return

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = MemmapConversationDataset(
        cache_path,
        cfg.model.context,
        num_rows,
    )

    if len(dataset) < 2:
        console.print(
            "[red]"
            "Not enough training examples."
            "[/red]"
        )
        return

    val_size = max(
        1,
        int(0.02 * len(dataset)),
    )

    val_size = min(
        val_size,
        len(dataset) - 1,
    )

    train_set, val_set = random_split(
        dataset,
        [
            len(dataset) - val_size,
            val_size,
        ],
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model, start_step, saved_optimizer_state = (
        load_model(
            cfg,
            tokenizer,
        )
    )

    model.to(device)

    if is_distributed():
        init_distributed()
        model = wrap_model(model, device)

    # --------------------------------------------------------
    # AUTOMATIC BIG BENCHMARK
    # --------------------------------------------------------

    if device.type == "cpu":
        big_autotune(
            cfg,
            model,
            train_set,
            device,
        )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = build_dataloader(
        train_set,
        cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        device=device,
    )

    val_loader = build_dataloader(
        val_set,
        cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        device=device,
    )

    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model = maybe_compile(
        model,
        device,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    if saved_optimizer_state is not None:
        try:
            optimizer.load_state_dict(
                saved_optimizer_state
            )

            console.print(
                "[green]"
                "Optimizer state restored."
                "[/green]"
            )

        except Exception as exc:
            console.print(
                f"[yellow]"
                f"Could not restore optimizer state: "
                f"{exc}. Starting optimizer fresh."
                f"[/yellow]"
            )

    warmup_steps_lr = max(1, int(cfg.train.warmup_steps))
    max_steps_lr = max(1, int(cfg.train.max_steps))
    peak_lr = float(cfg.train.lr)
    min_lr = peak_lr * 0.1

    def lr_at_step(s: int) -> float:
        if s < warmup_steps_lr:
            return peak_lr * (s + 1) / warmup_steps_lr
        progress = (s - warmup_steps_lr) / max(1, max_steps_lr - warmup_steps_lr)
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (peak_lr - min_lr) * coeff

    def apply_lr(s: int) -> None:
        current_lr = lr_at_step(s)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    model_for_info = model

    if hasattr(
        model,
        "_orig_mod",
    ):
        model_for_info = model._orig_mod

    hf_repo_id = getattr(cfg.train, "hf_repo_id", DEFAULT_HF_REPO_ID)

    console.print(
        f"[green]"
        f"Model params: "
        f"{model_for_info.num_params():,}"
        f"[/green]"
    )

    console.print(
        f"[green]"
        f"Training examples: "
        f"{len(train_set):,}"
        f"[/green]"
    )

    console.print(
        f"[green]"
        f"Validation examples: "
        f"{len(val_set):,}"
        f"[/green]"
    )

    console.print(
        f"[green]"
        f"Batch size: "
        f"{cfg.train.batch_size}"
        f"[/green]"
    )

    console.print(
        f"[green]"
        f"Gradient accumulation: "
        f"{cfg.train.grad_accum}"
        f"[/green]"
    )

    console.print(
        f"[green]"
        f"Hugging Face Auto-upload: "
        f"{hf_repo_id}"
        f"[/green]"
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    step = start_step
    skipped = 0

    data_iter = iter(
        train_loader
    )

    grad_accum = max(
        1,
        int(cfg.train.grad_accum),
    )

    max_steps = int(
        cfg.train.max_steps
    )

    checkpoint_every = max(
        1,
        int(cfg.train.checkpoint_every),
    )

    upload_every = max(
        1,
        int(getattr(cfg.train, "upload_every", checkpoint_every * 10)),
    )

    eval_every = max(
        1,
        int(cfg.train.eval_every),
    )

    grad_clip = float(
        cfg.train.grad_clip
    )

    accumulation_counter = 0
    tokens_seen = 0

    training_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    import select as _select
    _compact = {"on": False}

    def _check_key():
        if _select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line == "s":
                _compact["on"] = not _compact["on"]

    with Progress(
        TextColumn(
            "[bold blue]"
            "step {task.completed}/{task.total}"
        ),
        BarColumn(),
        TextColumn(
            "{task.fields[extra]}"
        ),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(
            "train",
            total=max_steps,
            loss=0.0,
            skipped=0,
            tokens_per_sec=0.0,
            steps_per_min=0.0,
            extra="",
        )

        progress.update(
            task,
            completed=min(
                start_step,
                max_steps,
            ),
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------

        while step < max_steps:

            try:
                x, y = next(
                    data_iter
                )

            except StopIteration:
                data_iter = iter(
                    train_loader
                )

                x, y = next(
                    data_iter
                )

            x, y = move_batch(
                x,
                y,
                device,
            )

            tokens_seen += int(
                x.numel()
            )

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            try:
                with get_autocast_context(
                    device
                ):
                    _, loss = model(
                        x,
                        y,
                    )

            except RuntimeError as exc:

                if (
                    device.type == "cuda"
                    and "out of memory"
                    in str(exc).lower()
                ):
                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    torch.cuda.empty_cache()

                    raise RuntimeError(
                        "CUDA ran out of memory. "
                        "Reduce train.batch_size."
                    ) from exc

                raise

            # ------------------------------------------------
            # NAN / INF PROTECTION
            # ------------------------------------------------

            # MPS: keep the training path asynchronous.
            # Only synchronize for numerical validation periodically.
            loss_invalid = False
            if step % 100 == 0:
                loss_invalid = not bool(
                    torch.isfinite(loss.detach()).all().cpu().item()
                )

            if loss_invalid:
                skipped += 1

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulation_counter = 0

                progress.update(
                    task,
                    skipped=skipped,
                )

                continue

            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            (
                loss / grad_accum
            ).backward()

            accumulation_counter += 1

            # ------------------------------------------------
            # OPTIMIZER
            # ------------------------------------------------

            if (
                accumulation_counter
                >= grad_accum
            ):

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )
                )

                # MPS: only synchronize for gradient validation periodically.
                grad_invalid = False
                if step % 100 == 0:
                    grad_invalid = not bool(
                        torch.isfinite(grad_norm.detach()).all().cpu().item()
                    )

                if grad_invalid:
                    skipped += 1

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    accumulation_counter = 0

                    progress.update(
                        task,
                        skipped=skipped,
                    )

                    continue

                apply_lr(step)

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulation_counter = 0

            # ------------------------------------------------
            # STEP
            # ------------------------------------------------

            step += 1

            elapsed = max(
                time.perf_counter()
                - training_start,
                1e-9,
            )

            tokens_per_sec = (
                tokens_seen
                / elapsed
            )

            _check_key()
            pct = (step / max_steps) * 100 if max_steps else 0.0
            # Do not pull a scalar from MPS every step.
            # Sample the loss only every 100 steps.
            loss_value = float(
                    loss.detach().cpu().item()
                )


            if step % 10 == 0:
                loss_value = float(
                    loss.detach().cpu().item()
                )

            if _compact["on"]:
                extra_text = f"{pct:.1f}%"
            elif loss_value is None:
                extra_text = (
                    f"loss={loss_value} "
                    f"skipped={skipped} "
                    f"tok/s={tokens_per_sec:.0f} "
                    f"steps/min={(step / elapsed) * 60:.1f} "
                    f"{pct:.1f}%"
                )
            else:
                extra_text = (
                    f"loss={loss_value:.4f} "
                    f"skipped={skipped} "
                    f"tok/s={tokens_per_sec:.0f} "
                    f"steps/min={(step / elapsed) * 60:.1f} "
                    f"{pct:.1f}%"
                )

            progress_kwargs = {
                "completed": step,
                "skipped": skipped,
                "tokens_per_sec": tokens_per_sec,
                "steps_per_min": (step / elapsed) * 60,
                "extra": extra_text,
            }

            if loss_value is not None:
                progress_kwargs["loss"] = loss_value

            progress.update(
                task,
                **progress_kwargs,
            )

            # ------------------------------------------------
            # CHECKPOINT & AUTOUPLOAD
            # ------------------------------------------------

            if (
                step
                % checkpoint_every
                == 0
            ):
                ckpt_path = save_checkpoint(
                    cfg,
                    model,
                    optimizer,
                    step,
                    finite=True,
                )
                if step % upload_every == 0:
                    upload_to_huggingface(
                        cfg=cfg,
                        checkpoint_path=ckpt_path,
                        repo_id=hf_repo_id,
                        step=step,
                    )

            # ------------------------------------------------
            # EVALUATION
            # ------------------------------------------------

            if (
                step
                % eval_every
                == 0
            ):

                val_loss = evaluate(
                    model,
                    val_loader,
                    device,
                )

                console.print(
                    f"[yellow]"
                    f"step {step:,} "
                    f"val_loss={val_loss:.4f} "
                    f"tok/s={tokens_per_sec:.0f}"
                    f"[/yellow]"
                )

                model.train()

        # ----------------------------------------------------
        # FLUSH PARTIAL GRADIENTS
        # ----------------------------------------------------

        if accumulation_counter > 0:

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )
            )

            # Partial-gradient flush happens only at the end.
            # Avoid an MPS scalar synchronization here; step the optimizer.
            apply_lr(step)

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

    # --------------------------------------------------------
    # FINAL CHECKPOINT & AUTOUPLOAD
    # --------------------------------------------------------

    final_ckpt_path = save_checkpoint(
        cfg,
        model,
        optimizer,
        step,
        finite=True,
    )
    upload_to_huggingface(
        cfg=cfg,
        checkpoint_path=final_ckpt_path,
        repo_id=hf_repo_id,
        step=step,
    )

    elapsed = max(
        time.perf_counter()
        - training_start,
        1e-9,
    )

    tokens_per_sec = (
        tokens_seen
        / elapsed
    )

    console.print(
        f"[bold green]"
        f"Training complete. "
        f"Skipped {skipped} non-finite steps. "
        f"Average throughput: "
        f"{tokens_per_sec:.0f} tokens/sec."
        f"[/bold green]"
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.inference_mode()
def evaluate(
    model: TinyGPT,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Fast MPS-safe validation.

    Only evaluates a bounded number of batches and keeps the loss
    accumulation on the device. This avoids thousands of MPS -> CPU
    synchronizations during validation.
    """

    model.eval()

    # The complete validation set is ~216k examples.
    # With batch size 4 that can be ~54k forward passes.
    # That is far too expensive to run every 250 training steps.
    max_batches = 200

    total_loss = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )

    count = 0

    for x, y in loader:
        if count >= max_batches:
            break

        x, y = move_batch(
            x,
            y,
            device,
        )

        try:
            with get_autocast_context(device):
                _, loss = model(
                    x,
                    y,
                )

        except RuntimeError as exc:
            if (
                device.type == "cuda"
                and "out of memory" in str(exc).lower()
            ):
                torch.cuda.empty_cache()

                console.print(
                    "[yellow]"
                    "CUDA ran out of memory during validation."
                    "[/yellow]"
                )

                continue

            raise

        # Keep everything on MPS.
        # Do NOT use bool(), .item(), or torch.isfinite() here.
        total_loss = total_loss + loss.detach().float()
        count += 1

    if count == 0:
        return float("nan")

    # Exactly ONE synchronization for the entire evaluation.
    return float(
        (total_loss / count).cpu().item()
    )
