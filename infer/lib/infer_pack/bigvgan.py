from pathlib import Path
from typing import Sequence, cast
import types

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F
from huggingface_hub import hf_hub_download


def _load_bigvgan() -> types.ModuleType:
    try:
        import bigvgan
    except ImportError as exc:
        raise ImportError(
            "BigVGAN is required for v3 models. Install dependencies with `uv sync`."
        ) from exc
    return bigvgan


def _load_pretrained_bigvgan(model_id: str, use_cuda_kernel: bool) -> nn.Module:
    bigvgan = _load_bigvgan()
    config_path = Path(hf_hub_download(repo_id=model_id, filename="config.json"))
    weights_path = Path(
        hf_hub_download(repo_id=model_id, filename="bigvgan_generator.pt")
    )
    h = bigvgan.bigvgan.load_hparams_from_json(config_path)
    model = bigvgan.BigVGAN(h, use_cuda_kernel=use_cuda_kernel)
    checkpoint_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(checkpoint_dict["generator"])
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(checkpoint_dict["generator"])
    return cast(nn.Module, model)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            padding=dilation,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return x + self.conv(self.act(x))


class BigVGANMelDecoder(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        hidden_channels: int,
        num_mels: int,
        gin_channels: int,
    ) -> None:
        super().__init__()
        self.input_proj = Conv1d(initial_channel, hidden_channels, 1)
        self.pitch_proj = Conv1d(1, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(hidden_channels, dilation) for dilation in (1, 3, 9, 27)]
        )
        self.output_proj = Conv1d(hidden_channels, num_mels, 1)
        self.cond = Conv1d(gin_channels, hidden_channels, 1) if gin_channels else None
        self.act = nn.SiLU()

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        pitch = torch.log1p(torch.clamp(f0, min=0.0)).unsqueeze(1)
        x = self.input_proj(x) + self.pitch_proj(pitch)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.act(x))


class BigVGANNSFGenerator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: object,
        resblock_kernel_sizes: Sequence[int],
        resblock_dilation_sizes: Sequence[Sequence[int]],
        upsample_rates: Sequence[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Sequence[int],
        gin_channels: int,
        sr: str | int,
        is_half: bool = False,
        model_id: str = "nvidia/bigvgan_v2_44khz_128band_512x",
        use_cuda_kernel: bool = False,
    ) -> None:
        super().__init__()
        del resblock
        del resblock_kernel_sizes
        del resblock_dilation_sizes
        del upsample_rates
        del upsample_kernel_sizes
        del is_half
        if isinstance(sr, str) and sr != "44k":
            raise ValueError("v3 BigVGAN generator currently supports only 44k.")
        if isinstance(sr, int) and sr != 44100:
            raise ValueError("v3 BigVGAN generator currently supports only 44100 Hz.")

        self.bigvgan = _load_pretrained_bigvgan(
            model_id=model_id, use_cuda_kernel=use_cuda_kernel
        )
        # type: ignore[not-callable]
        # self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        for parameter in self.bigvgan.parameters():
            parameter.requires_grad_(False)

        h = self.bigvgan.h
        self.mel_decoder = BigVGANMelDecoder(
            initial_channel=initial_channel,
            hidden_channels=upsample_initial_channel,
            num_mels=int(getattr(h, "num_mels")),
            gin_channels=gin_channels,
        )
        self.upp = int(getattr(h, "hop_size", 512))

    @property
    def sampling_rate(self) -> int:
        return int(getattr(self.bigvgan.h, "sampling_rate"))

    def remove_weight_norm(self) -> None:
        # type: ignore[operator]
        self.bigvgan.remove_weight_norm()

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: torch.Tensor | None = None,
        n_res: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if n_res is not None:
            n = int(n_res.item())
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")
            if n != f0.shape[-1]:
                f0 = F.interpolate(f0.unsqueeze(1), size=n, mode="linear").squeeze(1)

        mel = self.mel_decoder(x, f0, g=g)
        audio = self.bigvgan(mel.float())
        expected_samples = mel.shape[-1] * self.upp
        if audio.shape[-1] != expected_samples:
            audio = F.interpolate(audio, size=expected_samples, mode="linear")
        return torch.tanh(audio)
