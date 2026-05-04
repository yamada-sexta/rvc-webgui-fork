from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class V2TrainConfig:
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
    init_lr_ratio: int
    warmup_epochs: int
    c_mel: int
    c_kl: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V2TrainConfig:
        if isinstance(data.get("betas"), list):
            data["betas"] = tuple(data["betas"])
        return cls(**data)


@dataclass(frozen=True)
class V2DataConfig:
    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V2DataConfig:
        return cls(**data)


@dataclass(frozen=True)
class V2ModelConfig:
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: int
    resblock: str
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    upsample_rates: list[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]
    use_spectral_norm: bool
    gin_channels: int
    spk_embed_dim: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V2ModelConfig:
        return cls(**data)


@dataclass(frozen=True)
class V2TrainingConfig:
    train: V2TrainConfig
    data: V2DataConfig
    model: V2ModelConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V2TrainingConfig:
        return cls(
            train=V2TrainConfig.from_dict(data["train"]),
            data=V2DataConfig.from_dict(data["data"]),
            model=V2ModelConfig.from_dict(data["model"]),
        )

