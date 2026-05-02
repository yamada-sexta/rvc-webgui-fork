from dataclasses import asdict, dataclass, field

from lib.json_validation import TrainingConfig


@dataclass(frozen=True)
class V3TrainConfig:
    log_interval: int = 200
    seed: int = 1234
    epochs: int = 20000
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    batch_size: int = 4
    fp16_run: bool = True
    lr_decay: float = 0.999875
    segment_size: int = 32768
    init_lr_ratio: int = 1
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0


@dataclass(frozen=True)
class V3DataConfig:
    max_wav_value: float = 32768.0
    sampling_rate: int = 44100
    filter_length: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float | None = 22050.0


@dataclass(frozen=True)
class V3ModelConfig:
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: int = 0
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4, 4])
    use_spectral_norm: bool = False
    gin_channels: int = 256
    spk_embed_dim: int = 109


@dataclass(frozen=True)
class V3TrainingConfig:
    train: V3TrainConfig = field(default_factory=V3TrainConfig)
    data: V3DataConfig = field(default_factory=V3DataConfig)
    model: V3ModelConfig = field(default_factory=V3ModelConfig)


def get_v3_training_config() -> TrainingConfig:
    return TrainingConfig.model_validate(asdict(V3TrainingConfig()))
