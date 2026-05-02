import os
import sys
import json
import shutil
from multiprocessing import cpu_count
from functools import wraps
from typing import Literal, TypeAlias, TypeVar, cast

from tap import Tap
from loguru import logger

from lib.accelerate_utils import get_accelerator, use_half_precision
from lib.json_validation import TrainingConfig

VersionConfigPath: TypeAlias = Literal["v2/48k.json", "v2/32k.json"]

version_config_list: list[VersionConfigPath] = [
    "v2/48k.json",
    "v2/32k.json",
]

T = TypeVar("T")


VersionConfig = TrainingConfig


class ConfigArgs(Tap):
    # Listen port.
    port: int = 7865
    # Python command used for subprocess workers.
    pycmd: str = sys.executable or "python"
    # Launch in colab.
    colab: bool = False
    # Disable parallel processing.
    noparallel: bool = False
    # Do not open in browser automatically.
    noautoopen: bool = False


_singleton_instances: dict[type[object], object] = {}


def singleton_class(cls: type[T]) -> type[T]:
    @wraps(cls)
    def wrapper(*args: object, **kwargs: object) -> T:
        if cls not in _singleton_instances:
            _singleton_instances[cls] = cls(*args, **kwargs)
        return cast(T, _singleton_instances[cls])

    return cast(type[T], wrapper)


# def singleton_variable(func):
#     def wrapper(*args, **kwargs):
#         if not wrapper.instance:
#             wrapper.instance = func(*args, **kwargs)
#         return wrapper.instance

#     wrapper.instance = None
#     return wrapper


@singleton_class
class Config:
    use_jit: bool
    n_cpu: int
    gpu_name: str | None
    json_config: dict[str, VersionConfig]
    gpu_mem: int | None

    python_cmd: str
    listen_port: int
    iscolab: bool
    noparallel: bool
    noautoopen: bool

    instead: str
    preprocess_per: float
    x_pad: int
    x_query: int
    x_center: int
    x_max: int

    def __init__(self):
        accelerator = get_accelerator()
        self.use_jit: bool = False
        self.n_cpu: int = 0
        self.gpu_name: str | None = None
        self.json_config: dict[str, VersionConfig] = self.load_config_json()
        self.gpu_mem: int | None = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
        ) = self.arg_parse()
        self.instead: str = ""
        self.preprocess_per: float = 3.7
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict[str, VersionConfig]:
        d: dict[str, VersionConfig] = {}
        for config_file in version_config_list:
            p = f"configs/inuse/{config_file}"
            if not os.path.exists(p):
                shutil.copy(f"configs/{config_file}", p)
            with open(f"configs/inuse/{config_file}", "r") as f:
                d[config_file] = TrainingConfig.model_validate(json.load(f))
        return d

    @staticmethod
    def arg_parse() -> tuple[str, int, bool, bool, bool]:
        cmd_opts = ConfigArgs().parse_args()
        port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
        )

    def device_config(self) -> tuple:
        accelerator = get_accelerator()
        device = accelerator.device
        if device.type != "cpu":
            self.gpu_name = accelerator.state.device.type
            logger.info(f"Using Accelerate device {device}")
            if device.type != "cuda":
                self.gpu_mem = None
        else:
            logger.info("Accelerate selected CPU")
            self.instead = "cpu"

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

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

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.instead:
            logger.info(f"Use {self.instead} instead")
        logger.info(f"Half-precision floating-point: {is_half}, device: {device}")
        return x_pad, x_query, x_center, x_max
