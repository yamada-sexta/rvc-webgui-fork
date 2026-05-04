import json
from configs.v2_config import V2TrainingConfig
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from torch import nn
from scipy.io.wavfile import read
from tap import Tap
from loguru import logger

from configs.v3_config import get_v3_training_config
from lib.json_validation import ModelVersion, SampleRateName, TrainingConfig

# MATPLOTLIB_FLAG = False


class TrainArgs(Tap):
    # Checkpoint save frequency in epochs.
    save_every_epoch: int
    # Total training epochs.
    total_epoch: int
    # Pretrained generator path.
    pretrainG: Path | None = None
    # Pretrained discriminator path.
    pretrainD: Path | None = None
    # Training batch size.
    batch_size: int
    # Experiment directory name under logs.
    experiment_dir: str
    # Sample rate, such as 32k, 44k, or 48k.
    sample_rate: SampleRateName
    # Save extracted model weights when saving checkpoints.
    save_every_weights: Literal["0", "1"] = "0"
    # Model version.
    version: ModelVersion
    if_f0: Literal[0, 1]  # Whether to use f0 as an input, 1 or 0.

    if_latest: Literal[0, 1]  # Whether to save only the latest G/D pth files, 1 or 0.

    def configure(self) -> None:
        self.add_argument("-se", "--save_every_epoch")
        self.add_argument("-te", "--total_epoch")
        self.add_argument("-pg", "--pretrainG")
        self.add_argument("-pd", "--pretrainD")
        self.add_argument("-bs", "--batch_size")
        self.add_argument("-e", "--experiment_dir")
        self.add_argument("-sr", "--sample_rate")
        self.add_argument("-sw", "--save_every_weights")
        self.add_argument("-v", "--version")
        self.add_argument("-f0", "--if_f0")
        self.add_argument("-l", "--if_latest")


def load_checkpoint_d(
    checkpoint_path: Path,
    combd: nn.Module,
    sbd: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    load_opt: int = 1,
):
    assert checkpoint_path.is_file()
    checkpoint_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    ##################
    def go(model: nn.Module, bkey: str) -> nn.Module:
        saved_state_dict = checkpoint_dict[bkey]
        if hasattr(model, "module"):
            # state_dict = model.module.state_dict()
            raise NotImplementedError
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():  # Shape required by the model
            try:
                new_state_dict[k] = saved_state_dict[k]
                if saved_state_dict[k].shape != state_dict[k].shape:
                    logger.warning(
                        f"Shape mismatch for {k}. Need {state_dict[k].shape}, got {saved_state_dict[k].shape}"
                    )  #
                    raise KeyError
            except Exception:
                logger.info(f"{k} is not in the checkpoint")
                new_state_dict[k] = v  # Random values provided by the model
        if hasattr(model, "module"):
            # model.module.load_state_dict(new_state_dict, strict=False)
            raise NotImplementedError
        else:
            model.load_state_dict(new_state_dict, strict=False)
        return model

    go(combd, "combd")
    model = go(sbd, "sbd")
    #############
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ### If it cannot load, and it's empty, reinitialize it. It might also affect the update of the lr schedule, so catch it in the outermost layer of the train file
        #   try:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    #   except:
    #     traceback.print_exc()
    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    load_opt: int = 1,
) -> tuple[nn.Module, torch.optim.Optimizer | None, float, int]:
    assert checkpoint_path.is_file()
    checkpoint_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        # state_dict = model.module.state_dict()
        raise ValueError(
            "Model wrapped in DataParallel or DistributedDataParallel is not supported yet."
        )
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # Shape required by the model
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning(
                    f"Shape mismatch for {k}. Need {state_dict[k].shape}, got {saved_state_dict[k].shape}"
                )  #
                raise KeyError
        except Exception:
            logger.info(f"{k} is not in the checkpoint")
            new_state_dict[k] = v  # Random values provided by the model
    if hasattr(model, "module"):
        # model.module.load_state_dict(new_state_dict, strict=False)
        raise ValueError(
            "Model wrapped in DataParallel or DistributedDataParallel is not supported yet."
        )
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ### If it cannot load, and it's empty, reinitialize it. It might also affect the update of the lr schedule, so catch it in the outermost layer of the train file
        #   try:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    #   except:
    #     traceback.print_exc()
    assert isinstance(learning_rate, float)
    assert isinstance(iteration, int)
    assert isinstance(model, nn.Module)
    assert optimizer is None or isinstance(optimizer, torch.optim.Optimizer)

    logger.info(f"Loaded checkpoint '{checkpoint_path}' (epoch {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path: Path):
    logger.info(
        f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}"
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def save_checkpoint_d(
    combd, sbd, optimizer, learning_rate: float, iteration, checkpoint_path: Path
):
    logger.info(
        f"Saving model and optimizer state at epoch {iteration} to {checkpoint_path}"
    )
    if hasattr(combd, "module"):
        state_dict_combd = combd.module.state_dict()
    else:
        state_dict_combd = combd.state_dict()
    if hasattr(sbd, "module"):
        state_dict_sbd = sbd.module.state_dict()
    else:
        state_dict_sbd = sbd.state_dict()
    torch.save(
        {
            "combd": state_dict_combd,
            "sbd": state_dict_sbd,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def latest_checkpoint_path(dir_path: Path, regex: str = "G_*.pth") -> Path:
    f_list = sorted(
        dir_path.glob(regex), key=lambda f: int("".join(filter(str.isdigit, f.name)))
    )
    x = f_list[-1]
    logger.debug(x)
    return x


def load_wav_to_torch(full_path: Path) -> tuple[torch.FloatTensor, int]:
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(
    filename: Path, split: str = "|"
) -> list[tuple[Path, Path, Path, Path, str]]:
    try:
        with open(filename, encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(filename) as f:
            lines = f.readlines()

    res: list[tuple[Path, Path, Path, Path, str]] = []
    for line in lines:
        parts = line.strip().split(split)
        if len(parts) != 5:
            raise ValueError(
                f"Expected 5 pipe-separated fields (audiopath|phone|pitch|pitchf|dv) "
                f"in {filename}, got {len(parts)}: {line.strip()!r}"
            )
        res.append(
            (Path(parts[0]), Path(parts[1]), Path(parts[2]), Path(parts[3]), parts[4])
        )

    return res


def get_hparams() -> "HParams":
    """
    todo:
      The ending group of seven:
        Save frequency, total epochs                    done
        bs                                    done
        pretrainG、pretrainD                  done
        Accelerator selection                  done
        if_latest                             done
      Model: if_f0                             done
      Sample rate: Auto-select config                  done
      -m:
        Auto-determine training_files path, change hps.data.training_files in train_nsf_load_pretrain.py    done
      -c is no longer needed
    """
    args = TrainArgs().parse_args()
    name = args.experiment_dir
    experiment_dir = Path("./logs") / args.experiment_dir

    if args.version == "v3":
        config = get_v3_training_config()

    elif args.version == "v2":
        config_path = experiment_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config file not found at {config_path}. Please ensure the config file is present."
            )
        config = V2TrainingConfig.from_dict(json.loads(config_path.read_text()))
    else:
        raise ValueError(f"Unsupported model version: {args.version}")

    logger.info(f"Using training config: {config}")

    return HParams(
        train=TrainHParams(
            log_interval=config.train.log_interval,
            seed=config.train.seed,
            epochs=config.train.epochs,
            learning_rate=config.train.learning_rate,
            betas=config.train.betas,
            eps=config.train.eps,
            batch_size=args.batch_size,  # runtime overrides config
            fp16_run=config.train.fp16_run,
            lr_decay=config.train.lr_decay,
            segment_size=config.train.segment_size,
            init_lr_ratio=config.train.init_lr_ratio,
            warmup_epochs=config.train.warmup_epochs,
            c_mel=config.train.c_mel,
            c_kl=config.train.c_kl,
        ),
        data=DataHParams(
            max_wav_value=config.data.max_wav_value,
            sampling_rate=config.data.sampling_rate,
            filter_length=config.data.filter_length,
            hop_length=config.data.hop_length,
            win_length=config.data.win_length,
            n_mel_channels=config.data.n_mel_channels,
            mel_fmin=config.data.mel_fmin,
            mel_fmax=config.data.mel_fmax,
            training_files=experiment_dir / "filelist.txt",
        ),
        model=ModelHParams(
            inter_channels=config.model.inter_channels,
            hidden_channels=config.model.hidden_channels,
            filter_channels=config.model.filter_channels,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            kernel_size=config.model.kernel_size,
            p_dropout=float(config.model.p_dropout),
            resblock=config.model.resblock,
            resblock_kernel_sizes=tuple(config.model.resblock_kernel_sizes),
            resblock_dilation_sizes=tuple(
                tuple(item) for item in config.model.resblock_dilation_sizes
            ),
            upsample_rates=tuple(config.model.upsample_rates),
            upsample_initial_channel=config.model.upsample_initial_channel,
            upsample_kernel_sizes=tuple(config.model.upsample_kernel_sizes),
            use_spectral_norm=config.model.use_spectral_norm,
            gin_channels=config.model.gin_channels,
            spk_embed_dim=config.model.spk_embed_dim,
        ),
        model_dir=experiment_dir,  # runtime overrides config
        experiment_dir=experiment_dir,  # runtime overrides config
        save_every_epoch=args.save_every_epoch,  # runtime overrides config
        name=name,  # runtime overrides config
        total_epoch=args.total_epoch,  # runtime overrides config
        pretrainG=args.pretrainG,
        pretrainD=args.pretrainD,
        version=args.version,
        if_f0=args.if_f0,
        if_latest=args.if_latest,
        save_every_weights=args.save_every_weights,
        sample_rate=config.data.sampling_rate,  # runtime overrides config
    )


def check_git_hash(model_dir: Path) -> None:
    source_dir = Path(os.path.realpath(__file__)).parent
    git_check = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if git_check.returncode != 0:
        return

    cur_hash = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    git_hash_file = model_dir / "githash"
    if git_hash_file.exists():
        saved_hash = git_hash_file.read_text()
        if saved_hash != cur_hash:
            logger.warning(
                f"Git hash values are different. {saved_hash[:8]} (saved) != {cur_hash[:8]} (current)"
            )
    else:
        git_hash_file.write_text(cur_hash)


def get_logger(model_dir: Path, filename: str = "train.log", *, stdout: bool = False):
    log_dir = model_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    logger.remove()
    if stdout:
        logger.add(
            sys.stdout,
            level="INFO",
            serialize=True,
            enqueue=False,
            backtrace=False,
            diagnose=False,
        )
    logger.add(
        sys.stderr,
        level="INFO",
        serialize=False,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        log_path,
        level="INFO",
        serialize=True,
        enqueue=False,
        backtrace=False,
        diagnose=False,
    )
    return logger


def hparams_to_dict(value: "HParams") -> dict:
    if not isinstance(value, HParams):
        raise TypeError(f"Expected HParams instance, got {type(value)}")
        # return hparams_to_dict(asdict(value))
    # if isinstance(value, dict):
    #     return {str(key): hparams_to_dict(item) for key, item in value.items()}
    # if isinstance(value, (list, tuple)):
    #     return [hparams_to_dict(item) for item in value]
    # return value
    # return asdict(value)
    res = asdict(value)
    logger.debug(f"Converted HParams to dict: {res}")
    return res


@dataclass(frozen=True)
class TrainHParams:
    log_interval: int
    seed: int
    epochs: int
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    batch_size: int
    fp16_run: bool
    lr_decay: float
    segment_size: int
    init_lr_ratio: float
    warmup_epochs: int
    c_mel: float
    c_kl: float


@dataclass(frozen=True)
class DataHParams:
    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: float | None
    training_files: Path | None = None
    min_text_len: int = 1
    max_text_len: int = 5000


@dataclass(frozen=True)
class ModelHParams:
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_rates: tuple[int, ...]
    upsample_initial_channel: int
    upsample_kernel_sizes: tuple[int, ...]
    use_spectral_norm: bool
    gin_channels: int
    spk_embed_dim: int


@dataclass(frozen=True)
class HParams:
    train: TrainHParams
    data: DataHParams
    model: ModelHParams

    sample_rate: int
    model_dir: Path
    experiment_dir: Path
    save_every_epoch: int
    name: str
    total_epoch: int = 0
    pretrainG: Path | None = None
    pretrainD: Path | None = None
    version: ModelVersion = "v2"
    if_f0: Literal[0, 1] = 1
    if_latest: Literal[0, 1] = 0
    save_every_weights: Literal["0", "1"] = "0"
