import torch
import torch.nn as nn
from torch import Tensor
from typing import Final, cast

from .deepunet import DeepUnet

N_MELS: Final[int] = cast(int, getattr(nn, "N_MELS", 128))
N_CLASS: Final[int] = cast(int, getattr(nn, "N_CLASS", 360))


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super(E2E, self).__init__()

        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                self.BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel: Tensor) -> Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

    class BiGRU(nn.Module):
        def __init__(
            self,
            input_features: int,
            hidden_features: int,
            num_layers: int,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_features,
                hidden_features,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        def forward(self, x: Tensor) -> Tensor:
            return cast(Tensor, self.gru(x)[0])
