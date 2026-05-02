from pathlib import Path
from typing import Any, Literal, cast

import torch

from .layers.synthesizers import SynthesizerTrnMsNSFsid
from .jit import load_inputs, export_jit_model, save_pickle
from .types import (
    FileLike,
    JitRvcCheckpoint,
    RvcCheckpoint,
    synthesizer_config_args,
)


def get_synthesizer(
    cpt: RvcCheckpoint, device: int | str | torch.device = torch.device("cpu")
) -> tuple[SynthesizerTrnMsNSFsid, RvcCheckpoint]:
    config = cpt["config"]
    if not isinstance(config, list):
        raise ValueError("Only v2 list-config checkpoints can be loaded by get_synthesizer.")
    config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v2")
    if version != "v2" or if_f0 != 1:
        raise ValueError("Only v2 models with f0 are supported.")
    net_g = SynthesizerTrnMsNSFsid(
        *synthesizer_config_args(config),
        encoder_dim=768,
        use_f0=True,
    )
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


def load_synthesizer(
    pth_path: FileLike, device: int | str | torch.device = torch.device("cpu")
) -> tuple[SynthesizerTrnMsNSFsid, RvcCheckpoint]:
    return get_synthesizer(
        cast(
            RvcCheckpoint,
            torch.load(pth_path, map_location=torch.device("cpu"), weights_only=True),
        ),
        device,
    )


def synthesizer_jit_export(
    model_path: str | Path,
    mode: Literal["script", "trace"] = "script",
    inputs_path: str | Path | None = None,
    save_path: str | Path | None = None,
    device: str | torch.device = torch.device("cpu"),
    is_half: bool = False,
):
    model_path = Path(model_path)
    if not save_path:
        stem = model_path.with_suffix("")
        save_path = stem.with_suffix(".half.jit" if is_half else ".jit")
    else:
        save_path = Path(save_path)
    model, cpt = load_synthesizer(model_path, device)
    model.forward = model.infer
    inputs: dict[str, torch.Tensor] | None = None
    device_str = str(device)
    if mode == "trace":
        if inputs_path is None:
            raise ValueError("inputs_path is required when mode is 'trace'")
        inputs = load_inputs(inputs_path, device_str, is_half)
    ckpt = export_jit_model(model, mode, inputs, device, is_half)
    jit_cpt = cast(JitRvcCheckpoint, dict(cpt))
    jit_cpt["model"] = ckpt["model"]
    jit_cpt["device"] = device
    save_pickle(cast(dict[str, Any], dict(jit_cpt)), save_path)
    return jit_cpt
