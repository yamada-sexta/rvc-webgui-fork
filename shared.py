import logging
import os
import shutil
from pathlib import Path
from types import FrameType

import warnings
from dotenv import load_dotenv
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.bind(stdlib_logger=record.name).opt(
            depth=depth,
            exception=record.exc_info,
        ).log(level, record.getMessage())


def configure_startup_logging() -> None:
    logger.remove()
    logger.add(
        os.sys.stderr,
        level="INFO",
        backtrace=False,
        diagnose=False,
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)


load_dotenv()
configure_startup_logging()
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("torio").setLevel(logging.ERROR)
logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("git").setLevel(logging.INFO)
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import torch

from configs.config import Config
from i18n.i18n import I18nAuto
from infer.modules.vc.modules import VC
from lib.accelerate_utils import get_accelerator

now_dir = Path.cwd()
tmp = now_dir / "TEMP"
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(now_dir / "runtime/Lib/site-packages/infer_pack", ignore_errors=True)
shutil.rmtree(now_dir / "runtime/Lib/site-packages/uvr5_pack", ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)
(now_dir / "logs").mkdir(parents=True, exist_ok=True)
(now_dir / "assets/weights").mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(tmp)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config: Config = Config()
vc = VC(config)


i18n = I18nAuto()
logger.info(f"Use Language: {i18n}")
accelerator = get_accelerator()
logger.info(f"Accelerate device: {accelerator.device}")
default_batch_size = 1


weight_root = Path(os.getenv("WEIGHT_ROOT", "assets/weights"))
index_root = Path(os.getenv("INDEX_ROOT", "logs"))
outside_index_root = Path(os.getenv("OUTSIDE_INDEX_ROOT", "assets/indices"))
rmvpe_root = Path(os.getenv("RMVPE_ROOT", "assets/rmvpe"))

names = []
for entry in weight_root.iterdir():
    logger.debug(f"Checking weight candidate {entry.name}")
    if entry.suffix == ".pth":
        names.append(entry.name)
index_paths = [""]  # Fix for gradio 5


def lookup_indices(root: Path) -> None:
    # shared.index_paths
    for index_file in root.rglob("*.index"):
        if "trained" not in index_file.name:
            index_paths.append(str(index_file))


lookup_indices(index_root)
lookup_indices(outside_index_root)

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}
