import os
import sys
import traceback
from collections import OrderedDict
from pathlib import Path
from collections.abc import Sequence
from typing import Literal, Protocol, cast

import torch

from i18n.i18n import I18nAuto

i18n = I18nAuto()

type WeightMap = dict[str, torch.Tensor]
type CheckpointValue = WeightMap | list[object] | dict[str, object] | str | int
type CheckpointDict = OrderedDict[str, CheckpointValue]
type SampleRate = Literal["32k", "48k"]
type ModelVersion = Literal["v2", "v3"]


class HParamsData(Protocol):
    filter_length: int
    sampling_rate: int


class HParamsModel(Protocol):
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: Sequence[int]
    resblock_dilation_sizes: Sequence[Sequence[int]]
    upsample_rates: Sequence[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: Sequence[int]
    spk_embed_dim: int
    gin_channels: int
    dac_model_type: str
    dac_latent_dim: int
    transformer_layers: int
    transformer_ffn_channels: int


class SaveHParams(Protocol):
    data: HParamsData
    model: HParamsModel


def savee(
    ckpt: WeightMap,
    sr: SampleRate,
    if_f0: int | str,
    name: str,
    epoch: int,
    version: ModelVersion,
    hps: SaveHParams,
) -> str:
    if int(if_f0) != 1 or version not in {"v2", "v3"}:
        return "Only v2/v3 models with f0 are supported."
    try:
        weights: WeightMap = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            weights[key] = ckpt[key].half()
        opt: CheckpointDict = OrderedDict()
        opt["weight"] = weights
        if version == "v3":
            opt["config"] = {
                "architecture": "TransformerDacGenerator",
                "phone_channels": 768,
                "hidden_channels": hps.model.hidden_channels,
                "n_heads": hps.model.n_heads,
                "dac_model_type": hps.model.dac_model_type,
                "dac_latent_dim": hps.model.dac_latent_dim,
                "transformer_layers": hps.model.transformer_layers,
                "transformer_ffn_channels": hps.model.transformer_ffn_channels,
                "spk_embed_dim": hps.model.spk_embed_dim,
                "sampling_rate": hps.data.sampling_rate,
            }
        else:
            opt["config"] = [
                hps.data.filter_length // 2 + 1,
                32,
                hps.model.inter_channels,
                hps.model.hidden_channels,
                hps.model.filter_channels,
                hps.model.n_heads,
                hps.model.n_layers,
                hps.model.kernel_size,
                hps.model.p_dropout,
                hps.model.resblock,
                hps.model.resblock_kernel_sizes,
                hps.model.resblock_dilation_sizes,
                hps.model.upsample_rates,
                hps.model.upsample_initial_channel,
                hps.model.upsample_kernel_sizes,
                hps.model.spk_embed_dim,
                hps.model.gin_channels,
                hps.data.sampling_rate,
            ]
        opt["info"] = f"{epoch}epoch"
        opt["sr"] = sr
        opt["f0"] = 1
        opt["version"] = version
        torch.save(opt, Path("assets/weights") / f"{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()


def show_info(path: Path | str) -> str:
    try:
        a = torch.load(path, map_location="cpu", weights_only=False)
        return (
            f"Model info:{a.get('info', 'None')}\n"
            f"Sample rate:{a.get('sr', 'None')}\n"
            f"Does the model use pitch guidance:{a.get('f0', 'None')}\n"
            f"Version:{a.get('version', 'None')}"
        )
    except:
        return traceback.format_exc()


def extract_small_model(
    path: Path | str,
    name: str,
    sr: SampleRate,
    if_f0: int | str,
    info: str,
    version: ModelVersion,
) -> str:
    if int(if_f0) != 1 or version != "v2":
        return "Only v2 models with f0 are supported."
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        ckpt = cast(WeightMap, ckpt)
        weights: WeightMap = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            weights[key] = ckpt[key].half()
        opt: CheckpointDict = OrderedDict()
        opt["weight"] = weights
        if sr == "48k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [12, 10, 2, 2],
                512,
                [24, 20, 4, 4],
                109,
                256,
                48000,
            ]
        elif sr == "32k":
            opt["config"] = [
                513,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 8, 2, 2],
                512,
                [20, 16, 4, 4],
                109,
                256,
                32000,
            ]
        else:
            return "Only v2 32k and 48k models are supported."
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["version"] = "v2"
        opt["sr"] = sr
        opt["f0"] = 1
        torch.save(opt, Path("assets/weights") / f"{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()


def change_info(path: Path | str, info: str, name: str) -> str:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt["info"] = info
        if name == "":
            name = Path(path).name
        torch.save(ckpt, Path("assets/weights") / name)
        return "Success."
    except:
        return traceback.format_exc()


def merge(
    path1: Path | str,
    path2: Path | str,
    alpha1: float,
    sr: SampleRate,
    f0: str,
    info: str,
    name: str,
    version: ModelVersion,
) -> str:
    if f0 != i18n("Yes") or version != "v2":
        return "Only v2 models with f0 are supported."
    try:

        def extract(ckpt: dict[str, object]) -> CheckpointDict:
            a = ckpt["model"]
            a = cast(WeightMap, a)
            weights: WeightMap = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                weights[key] = a[key]
            opt: CheckpointDict = OrderedDict()
            opt["weight"] = weights
            return opt

        ckpt1 = torch.load(path1, map_location="cpu", weights_only=False)
        ckpt2 = torch.load(path2, map_location="cpu", weights_only=False)
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        weights: WeightMap = {}
        for key in ckpt1.keys():
            # try:
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                weights[key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                weights[key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        opt: CheckpointDict = OrderedDict()
        opt["weight"] = weights
        # except:
        #     pdb.set_trace()
        opt["config"] = cfg
        opt["sr"] = sr
        opt["f0"] = 1
        opt["version"] = "v2"
        opt["info"] = info
        torch.save(opt, Path("assets/weights") / f"{name}.pth")
        return "Success."
    except:
        return traceback.format_exc()
