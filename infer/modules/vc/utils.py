from pathlib import Path
from typing import List, Tuple, Union

from fairseq import checkpoint_utils
from configs.config import Config
from fairseq.models.hubert.hubert import HubertModel
import shared
from lib.accelerate_utils import get_device, use_half_precision




def load_hubert(config: Config) -> HubertModel:  # hubert_model is a torch.nn.Module
    models: List[HubertModel]

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(get_device())
    if use_half_precision():
        try:
            hubert_model = hubert_model.half()
        except Exception as e:
            print(
                "Warning: could not convert HuBERT to half — keeping float32. Error:",
                e,
            )
            hubert_model = hubert_model.float()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
