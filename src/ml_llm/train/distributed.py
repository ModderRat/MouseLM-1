from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return os.environ.get("ML_LLM_DISTRIBUTED", "0") == "1"


def init_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(
        backend="gloo",
        init_method=os.environ.get("ML_LLM_INIT_METHOD", "env://"),
        rank=rank,
        world_size=world_size,
    )

    return rank, world_size, local_rank


def wrap_model(model, device: torch.device):
    if not is_distributed():
        return model

    return torch.nn.parallel.DistributedDataParallel(model)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
