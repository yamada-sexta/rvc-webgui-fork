from functools import lru_cache
from dataclasses import dataclass
import os
import sys
import json
import shutil
from multiprocessing import cpu_count
from functools import wraps
from typing import Literal, TypeAlias, TypeVar, cast
from pathlib import Path
from tap import Tap
from loguru import logger

from lib.accelerate_utils import get_accelerator, use_half_precision
from configs.v2_config import (
    V2DataConfig,
    V2ModelConfig,
    V2TrainConfig,
    V2TrainingConfig,
)

VersionConfigPath: TypeAlias = Literal["v2/48k.json", "v2/32k.json"]

version_config_list: tuple[VersionConfigPath, ...] = (
    "v2/48k.json",
    "v2/32k.json",
)

T = TypeVar("T")


VersionConfig = V2TrainingConfig


class ConfigArgs(Tap):
    # Listen port.
    port: int = 7865
    # Python command used for subprocess workers.
    pycmd: str = sys.executable
    # Launch in colab.
    colab: bool = False
    # Disable parallel processing.
    noparallel: bool = False
    # Do not open in browser automatically.
    noautoopen: bool = False


@dataclass(frozen=True, slots=True)
class ConfigData:
    n_cpu: int
    gpu_name: str | None
    json_config: dict[str, VersionConfig]
    gpu_mem: int | None

    python_cmd: str
    listen_port: int
    is_colab: bool
    no_parallel: bool
    # no_auto_open: bool

    # instead: str
    preprocess_per: float
    x_pad: int
    x_query: int
    x_center: int
    x_max: int


def load_config_json() -> dict[str, VersionConfig]:
    d: dict[str, VersionConfig] = {}
    for config_file in version_config_list:
        # p = f"configs/inuse/{config_file}"
        path = Path("configs/inuse") / config_file
        if not path.exists():
            shutil.copy(f"configs/{config_file}", path)
        with open(path, "r") as f:
            data_dict = json.load(f)
            d[config_file] = V2TrainingConfig.from_dict(data_dict)
    return d


@lru_cache(maxsize=1)
def get_config() -> ConfigData:
    accelerator = get_accelerator()
    n_cpu: int = 0
    gpu_name: str | None = None
    json_config: dict[str, VersionConfig] = load_config_json()
    gpu_mem: int | None = None

    args = ConfigArgs().parse_args()

    instead: str = ""
    preprocess_per: float = 3.7
    # x_pad, x_query, x_center, x_max = device_config(accelerator)

    accelerator = get_accelerator()
    device = accelerator.device
    if device.type != "cpu":
        gpu_name = accelerator.state.device.type
        logger.info(f"Using Accelerate device {device}")
        if device.type != "cuda":
            gpu_mem = None
    else:
        logger.info("Accelerate selected CPU")
        instead = "cpu"
    if n_cpu == 0:
        n_cpu = cpu_count()
    is_half = use_half_precision()
    if is_half:
        # VRAM >= 6GB: use x_pad=3, x_query=10, x_center=60, x_max=65
        x_pad = 3
        x_query = 10
        x_center = 60
        x_max = 65
    else:
        # VRAM >= 4GB: use x_pad=1, x_query=6, x_center=38, x_max=41
        x_pad = 1
        x_query = 6
        x_center = 38
        x_max = 41
    if gpu_mem is not None and gpu_mem <= 4:
        x_pad = 1
        x_query = 5
        x_center = 30
        x_max = 32
    if instead:
        logger.info(f"Use {instead} instead")
    logger.info(f"Half-precision floating-point: {is_half}, device: {device}")

    return ConfigData(
        n_cpu=n_cpu,
        gpu_name=gpu_name,
        json_config=json_config,
        gpu_mem=gpu_mem,
        python_cmd=args.pycmd,
        listen_port=args.port,
        is_colab=args.colab,
        no_parallel=args.noparallel,
        # no_auto_open=args.noautoopen,
        # instead=instead,
        preprocess_per=preprocess_per,
        x_pad=x_pad,
        x_query=x_query,
        x_center=x_center,
        x_max=x_max,
    )
