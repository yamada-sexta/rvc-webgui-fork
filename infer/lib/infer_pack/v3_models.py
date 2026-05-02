from typing import Protocol, cast

import torch
from torch import nn
from torch.nn import functional as F

class DacEncoderOutputLike(Protocol):
    quantized_representation: torch.Tensor
    audio_codes: torch.Tensor


class DacDecoderOutputLike(Protocol):
    audio_values: torch.Tensor


class FrozenDacCodec(nn.Module):
    def __init__(self, sample_rate: int, model_id: str = "descript/dac_16khz") -> None:
        super().__init__()
        from transformers import DacModel

        self.sample_rate = sample_rate
        self.codec = DacModel.from_pretrained(model_id)
        self.codec_sample_rate = int(getattr(self.codec.config, "sampling_rate", 16000))
        self.hop_length = int(getattr(self.codec.config, "hop_length", 512))
        self.codec.eval()
        for parameter in self.codec.parameters():
            parameter.requires_grad_(False)

    def _resample_for_codec(self, audio: torch.Tensor) -> torch.Tensor:
        if self.sample_rate == self.codec_sample_rate:
            return audio
        target_length = max(1, round(audio.shape[-1] * self.codec_sample_rate / self.sample_rate))
        return F.interpolate(audio, size=target_length, mode="linear", align_corners=False)

    def encode_outputs(self, audio: torch.Tensor) -> DacEncoderOutputLike:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        audio = self._resample_for_codec(audio.clamp(-1.0, 1.0))
        return cast(DacEncoderOutputLike, self.codec.encode(audio))

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode_outputs(audio).quantized_representation

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        outputs = cast(DacDecoderOutputLike, self.codec.decode(latents))
        audio = outputs.audio_values
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def decode_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        outputs = cast(DacDecoderOutputLike, self.codec.decode(audio_codes=audio_codes))
        audio = outputs.audio_values
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio

    def logits_to_quantized(self, logits: torch.Tensor) -> torch.Tensor:
        quantized: torch.Tensor | float = 0.0
        probabilities = torch.softmax(logits.float(), dim=2).to(logits.dtype)
        for codebook_idx, quantizer in enumerate(self.codec.quantizer.quantizers):
            codebook = quantizer.codebook.weight.to(probabilities)
            embedded = torch.einsum(
                "bkt,ke->bet",
                probabilities[:, codebook_idx],
                codebook,
            )
            quantized = quantized + quantizer.out_proj(embedded)
        return cast(torch.Tensor, quantized)

    def to_audio_device(self, device: torch.device) -> None:
        if next(self.codec.parameters()).device != device:
            self.codec.to(device)


class TransformerDacGenerator(nn.Module):
    def __init__(
        self,
        phone_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        p_dropout: float,
        spk_embed_dim: int,
        gin_channels: int,
        dac_latent_dim: int,
        dac_num_codebooks: int = 12,
        dac_codebook_size: int = 1024,
    ) -> None:
        super().__init__()
        self.phone_proj = nn.Linear(phone_channels, hidden_channels)
        self.pitch_embed = nn.Embedding(256, hidden_channels)
        self.spk_embed = nn.Embedding(spk_embed_dim, hidden_channels)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=filter_channels,
            dropout=float(p_dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_channels)
        self.dac_num_codebooks = dac_num_codebooks
        self.dac_codebook_size = dac_codebook_size
        self.proj = nn.Linear(hidden_channels, dac_num_codebooks * dac_codebook_size)

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        spec: torch.Tensor | None,
        spec_lengths: torch.Tensor | None,
        sid: torch.Tensor,
        ids_slice: torch.Tensor | None = None,
        segment_frames: int | None = None,
        target_frames: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.phone_proj(phone) + self.pitch_embed(pitch)
        x = x + self.spk_embed(sid).unsqueeze(1)
        max_len = phone.shape[1]
        padding_mask = torch.arange(max_len, device=phone.device).unsqueeze(0) >= phone_lengths.unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        if target_frames is not None and x.shape[1] != target_frames:
            ratio = target_frames / float(max_len)
            new_lengths = torch.round(phone_lengths.float() * ratio).long()
            x = F.interpolate(
                x.transpose(1, 2),
                size=target_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            padding_mask = torch.arange(target_frames, device=x.device).unsqueeze(0) >= new_lengths.unsqueeze(1)
        logits = self.proj(self.norm(x))
        logits = logits.reshape(
            logits.shape[0],
            logits.shape[1],
            self.dac_num_codebooks,
            self.dac_codebook_size,
        ).permute(0, 2, 3, 1)
        logits = logits.masked_fill(padding_mask[:, None, None, :], 0.0)
        if ids_slice is not None and segment_frames is not None:
            flat_logits = logits.flatten(1, 2)
            segments = []
            mask_segments = []
            for batch_idx, start in enumerate(ids_slice):
                start_int = int(start.item())
                segments.append(flat_logits[batch_idx : batch_idx + 1, :, start_int : start_int + segment_frames])
                mask_segments.append(padding_mask[batch_idx : batch_idx + 1, start_int : start_int + segment_frames])
            logits = torch.cat(segments, dim=0).reshape(
                logits.shape[0],
                self.dac_num_codebooks,
                self.dac_codebook_size,
                segment_frames,
            )
            padding_mask = torch.cat(mask_segments, dim=0)
        return logits, padding_mask
