from collections.abc import Sequence
from typing import NotRequired, TypedDict, cast

import torch

type SynthesizerConfigValue = int | float | str | list[int] | list[list[int]] | None
type SynthesizerConfig = list[SynthesizerConfigValue]
type SynthesizerConfigArgs = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    str,
    Sequence[int],
    Sequence[Sequence[int]],
    Sequence[int],
    int,
    Sequence[int],
    int,
    int,
    str | int | None,
]
type SynthesizerConfigArgsWithSr = tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    str,
    Sequence[int],
    Sequence[Sequence[int]],
    Sequence[int],
    int,
    Sequence[int],
    int,
    int,
    str | int,
]
type WeightMap = dict[str, torch.Tensor]
type RvcVersion = str


class RvcCheckpoint(TypedDict):
    config: SynthesizerConfig
    weight: WeightMap
    f0: NotRequired[int]
    version: NotRequired[RvcVersion]


class JitRvcCheckpoint(RvcCheckpoint, total=False):
    model: bytes
    device: int | str | torch.device


def synthesizer_config_args(config: SynthesizerConfig) -> SynthesizerConfigArgs:
    return cast(SynthesizerConfigArgs, tuple(config))


def synthesizer_config_args_with_sr(
    config: SynthesizerConfig,
) -> SynthesizerConfigArgsWithSr:
    return cast(SynthesizerConfigArgsWithSr, tuple(config))


def synthesizer_target_sr(config: SynthesizerConfig) -> int:
    return cast(int, config[-1])
