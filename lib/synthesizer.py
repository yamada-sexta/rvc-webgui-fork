from pathlib import Path
from typing import cast

import torch

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs768BigVGANsid,
    SynthesizerTrnMs768NSFsid,
)
from .types import (
    FileLike,
    RvcCheckpoint,
    synthesizer_config_args_with_sr,
)


def get_synthesizer(
    cpt: RvcCheckpoint, device: int | str | torch.device = torch.device("cpu")
) -> tuple[SynthesizerTrnMs768NSFsid | SynthesizerTrnMs768BigVGANsid, RvcCheckpoint]:
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v2")
    if version not in {"v2", "v3"} or if_f0 != 1:
        raise ValueError("Only v2/v3 models with f0 are supported.")
    model_cls = (
        SynthesizerTrnMs768BigVGANsid if version == "v3" else SynthesizerTrnMs768NSFsid
    )
    net_g = model_cls(*synthesizer_config_args_with_sr(cpt["config"]), is_half=False)
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


def load_synthesizer(
    pth_path: FileLike, device: int | str | torch.device = torch.device("cpu")
) -> tuple[SynthesizerTrnMs768NSFsid | SynthesizerTrnMs768BigVGANsid, RvcCheckpoint]:
    return get_synthesizer(
        cast(
            RvcCheckpoint,
            torch.load(pth_path, map_location=torch.device("cpu"), weights_only=True),
        ),
        device,
    )
