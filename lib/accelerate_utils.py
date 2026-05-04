from functools import lru_cache

import torch
from accelerate import Accelerator


@lru_cache(maxsize=1)
def get_accelerator() -> Accelerator:
    return Accelerator()


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return get_accelerator().device


@lru_cache(maxsize=1)
def device_string() -> str:
    return str(get_device())


@lru_cache(maxsize=1)
def use_half_precision() -> bool:
    accelerator = get_accelerator()
    return accelerator.device.type != "cpu" and accelerator.mixed_precision == "fp16"


def empty_cache() -> None:
    get_accelerator().free_memory()
