from typing import Literal, TypeAlias
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, RootModel

SampleRateName: TypeAlias = Literal["32k", "44k", "48k"]
ModelVersion: TypeAlias = Literal["v2", "v3"]
LogEventName: TypeAlias = Literal[
    "feature_args",
    "feature_empty",
    "feature_finished",
    "f0_args",
    "f0_empty",
    "f0_failed",
    "preprocess_failed",
    "preprocess_finished",
    "preprocess_started",
    "train_finished",
    "train_hparams",
    "ui_progress",
]
LogDetailEventName: TypeAlias = Literal[
    "f0_failed",
    "f0_model_loading",
    "f0_processing",
    "f0_progress",
    "f0_started",
    "feature_failed",
    "feature_model_loaded",
    "feature_model_loading",
    "feature_processing",
    "feature_progress",
    "feature_started",
    "preprocess_file_done",
    "preprocess_file_failed",
    "train_epoch_complete",
    "train_progress",
    "train_setup",
    "train_started",
]


class TrainSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

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


class DataSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: float | None
    training_files: Path | None = None


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    upsample_rates: list[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]
    use_spectral_norm: bool
    gin_channels: int
    spk_embed_dim: int


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: TrainSection
    data: DataSection
    model: ModelSection


class LocaleMap(RootModel[dict[str, str]]):
    def get(self, key: str, default: str) -> str:
        return self.root.get(key, default)


class LogHParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    sample_rate: SampleRateName | None = None


class LogExtra(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: LogEventName | None = None
    detail_event: LogDetailEventName | None = None
    fraction: float = 0.0
    message: str = ""
    stage: str | None = None
    hparams: LogHParams = Field(default_factory=LogHParams)


class LogRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str = ""
    extra: LogExtra = Field(default_factory=LogExtra)


class JsonLogPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    record: LogRecord = Field(default_factory=LogRecord)
