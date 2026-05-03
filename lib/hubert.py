from pathlib import Path
import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig
from safetensors.torch import save_file, load_model
from lib.accelerate_utils import get_device

class HubertModelWrapper(nn.Module):
    def __init__(self, hf_model: HubertModel):
        super().__init__()
        self.model = hf_model
        
    def extract_features(self, source: torch.Tensor, padding_mask: torch.Tensor | None = None, output_layer: int = 12, **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
        # fairseq padding_mask is True for padding. Transformers attention_mask is 1 for NOT padding.
        attention_mask = None
        if padding_mask is not None:
            attention_mask = (~padding_mask).long()
            
        outputs = self.model(
            input_values=source,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # In fairseq, output features are shape (B, T, C). Transformers gives (B, T, C)
        # Hidden states: 0 is embedding, 1 to 12 are the transformer layers.
        # So output_layer=12 corresponds to hidden_states[12]
        feats = outputs.hidden_states[output_layer]
        
        # fairseq returns (features, padding_mask) or similar tuple in RVC
        # Let's match the original return type of (feature, padding_mask)
        return feats, padding_mask
        
    def infer(self, source: torch.Tensor, padding_mask: torch.Tensor | None, output_layer: torch.Tensor | int) -> torch.Tensor:
        if isinstance(output_layer, torch.Tensor):
            output_layer_id = int(output_layer.item())
        else:
            output_layer_id = int(output_layer)
            
        if output_layer_id not in [9, 12]:
            raise ValueError(f"Only HuBERT output_layer=9 or 12 is supported. Got {output_layer_id}")
            
        logits, _ = self.extract_features(source=source, padding_mask=padding_mask, output_layer=output_layer_id)
        return logits

def convert_fairseq_to_hf(model_path: Path, safetensors_path: Path):
    import sys
    import types
    if 'fairseq' not in sys.modules:
        sys.modules['fairseq'] = types.ModuleType('fairseq')
        sys.modules['fairseq.data'] = types.ModuleType('fairseq.data')
        sys.modules['fairseq.data.dictionary'] = types.ModuleType('fairseq.data.dictionary')
        class DummyDict: pass
        setattr(sys.modules['fairseq.data.dictionary'], 'Dictionary', DummyDict)
        
    ckpt = torch.load(model_path, weights_only=False)
    fairseq_dict = ckpt['model']
    
    hf_config = HubertConfig()
    hf_model = HubertModel(hf_config)
    
    mapping = {
        "post_extract_proj": "feature_projection.projection",
        "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
        "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
        "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
        "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
        "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
        "self_attn_layer_norm": "encoder.layers.*.layer_norm",
        "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
        "fc2": "encoder.layers.*.feed_forward.output_dense",
        "final_layer_norm": "encoder.layers.*.final_layer_norm",
        "encoder.layer_norm": "encoder.layer_norm",
        "layer_norm": "feature_projection.layer_norm",
        "w2v_model.layer_norm": "feature_projection.layer_norm",
        "mask_emb": "masked_spec_embed",
    }
    
    hf_dict = hf_model.state_dict()
    new_dict = {}
    
    for name, value in fairseq_dict.items():
        if "conv_layers" in name:
            parts = name.split(".")
            layer_idx = int(parts[2])
            type_idx = int(parts[3])
            weight_type = parts[4]
            
            if type_idx == 0:
                mapped = f"feature_extractor.conv_layers.{layer_idx}.conv.{weight_type}"
            elif type_idx == 2:
                mapped = f"feature_extractor.conv_layers.{layer_idx}.layer_norm.{weight_type}"
            else:
                continue
            new_dict[mapped] = value
        else:
            for k, v in mapping.items():
                if k in name:
                    if "*" in v:
                        layer_idx = name.split(k)[0].split(".")[-2]
                        v = v.replace("*", layer_idx)
                    
                    weight_type = name.split(".")[-1]
                    if weight_type == "weight_g":
                        mapped = f"{v}.parametrizations.weight.original0"
                    elif weight_type == "weight_v":
                        mapped = f"{v}.parametrizations.weight.original1"
                    elif weight_type in ["weight", "bias"]:
                        mapped = f"{v}.{weight_type}"
                    else:
                        mapped = v
                        
                    new_dict[mapped] = value
                    break

    for k, v in new_dict.items():
        if k in hf_dict:
            if hf_dict[k].shape == v.shape:
                hf_dict[k] = v
                
    save_file(hf_dict, safetensors_path)

def get_hubert(
    model_path: Path = Path("assets/hubert/hubert_base.pt"),
    device: torch.device | None = None,
) -> HubertModelWrapper:
    if device is None:
        device = get_device()
        
    safetensors_path = model_path.with_suffix(".safetensors")
    
    if not safetensors_path.exists():
        print(f"Converting HuBERT from {model_path} to {safetensors_path}...")
        convert_fairseq_to_hf(model_path, safetensors_path)
        
    hf_config = HubertConfig()
    hf_model = HubertModel(hf_config)
    load_model(hf_model, safetensors_path)
    
    hf_model = hf_model.to(device)
    wrapper = HubertModelWrapper(hf_model)
    return wrapper.eval()
