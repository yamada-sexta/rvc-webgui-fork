from pathlib import Path
from typing import List, Tuple, Union

from configs.config import Config
import shared
from lib.accelerate_utils import get_device, use_half_precision




from lib.hubert import get_hubert

def load_hubert(config: Config):
    hubert_model = get_hubert()
    
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
    return hubert_model
