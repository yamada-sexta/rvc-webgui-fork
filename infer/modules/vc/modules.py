from lib.f0 import PitchMethod
import traceback
from pathlib import Path
from typing import Mapping, cast
import gradio as gr
import resampy
from loguru import logger

from configs.config import Config
from lib.types import (
    RvcCheckpoint,
    synthesizer_config_args_with_sr,
    synthesizer_target_sr,
)

import numpy as np
import torch
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs768BigVGANsid,
    SynthesizerTrnMs768NSFsid,
)
from infer.modules.vc.pipeline import Pipeline
from infer.modules.vc.utils import *
from lib.accelerate_utils import empty_cache, get_device, use_half_precision


def resample_audio(
    audio_array: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    # Check if the audio is stereo and downmix to mono
    if audio_array.ndim > 1 and audio_array.shape[1] > 1:
        # print("Detected stereo audio, downmixing to mono.")
        # Average the channels to create a mono signal
        audio_mono = audio_array.mean(axis=1)
    else:
        # Already mono or 1D array
        audio_mono = audio_array.flatten()  # Ensure it's 1D in case it's (N, 1)

    # print(f"Mono audio shape after downmixing: {audio_mono.shape}")

    if audio_mono.size < 10:  # A reasonable minimum length for resampling
        raise ValueError(
            f"Mono audio signal length ({audio_mono.size}) is too small to resample from {orig_sr} to {target_sr}. "
            "Ensure the audio file contains actual sound data."
        )

    # Perform resampling on the mono signal
    resampled_audio = resampy.resample(audio_mono, orig_sr, target_sr)
    # print(f"Resampled audio shape: {resampled_audio.shape}")
    return resampled_audio


class VC:
    def __init__(self: "VC", config: Config):
        # self.config = config
        self.n_spk: int | None = None
        self.tgt_sr: int | None = None
        self.net_g: SynthesizerTrnMs768NSFsid | SynthesizerTrnMs768BigVGANsid | None = (
            None
        )
        self.pipeline: Pipeline | None = None
        self.cpt: RvcCheckpoint | None = None
        self.version: str = "UNKNOWN"
        from lib.hubert import HubertModelWrapper
        self.hubert_model: HubertModelWrapper | None = None
        self.config: Config = config

    def _build_synthesizer(
        self, cpt: RvcCheckpoint
    ) -> SynthesizerTrnMs768NSFsid | SynthesizerTrnMs768BigVGANsid:
        version = cpt.get("version", "v2")
        model_cls = (
            SynthesizerTrnMs768BigVGANsid
            if version == "v3"
            else SynthesizerTrnMs768NSFsid
        )
        return model_cls(
            *synthesizer_config_args_with_sr(cpt["config"]),
            is_half=use_half_precision(),
        )

    def get_vc(
        self: "VC", sid: str | None, *to_return_protect: float
    ) -> Mapping[str, object]:
        logger.info(f"get_vc sid: {sid}, input protect: {to_return_protect}")
        if sid is None or sid == "":
            val = to_return_protect[0] if to_return_protect else 0.33
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, dict):
                val = val.get("value", 0.33)
            logger.info(f"No SID, returning protect: {val}")
            return {"visible": True, "value": val, "__type__": "update"}
        # self.pipeline
        logger.info(f"Get sid: {sid}")

        val = to_return_protect[0] if to_return_protect else 0.33
        if isinstance(val, (list, tuple)):
            val = val[0]
        if isinstance(val, dict):
            val = val.get("value", 0.33)

        to_return_protect0 = {
            "visible": True,
            "value": val,
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if self.hubert_model is not None:
                # Considering polling, we need to add a check to see if sid switched from having a model to not having one
                logger.info("Clean model cache")
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                empty_cache()
                # You just have to follow this instruction to clear it.
                cpt = self.cpt
                if cpt is not None:
                    self.version = cpt.get("version", "v2")
                    if self.version in {"v2", "v3"} and cpt.get("f0", 1) == 1:
                        self.net_g = self._build_synthesizer(cpt)
                self.net_g = None
                self.cpt = None
                empty_cache()
            return to_return_protect0
        person = shared.weight_root / sid
        logger.info(f"Loading: {person}")

        self.cpt = cast(
            RvcCheckpoint, torch.load(person, map_location="cpu", weights_only=False)
        )
        self.tgt_sr = synthesizer_target_sr(self.cpt["config"])
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.version = self.cpt.get("version", "v2")
        if self.version not in {"v2", "v3"} or self.cpt.get("f0", 1) != 1:
            raise ValueError("Only v2/v3 models with f0 are supported.")

        self.net_g = self._build_synthesizer(self.cpt)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(get_device())
        if use_half_precision():
            try:
                self.net_g = self.net_g.half()
            except Exception as e:
                self.net_g = self.net_g.float()
                print(
                    "Warning: could not convert model to half — keeping float32. Error:",
                    e,
                )
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(synthesizer_target_sr(self.cpt["config"]), self.config)
        # n_spk = self.cpt["config"][-3]
        res = to_return_protect0
        logger.info(f"Result {res}")

        return res

    def vc_single(
        self: "VC",
        sr_and_audio: tuple[int, np.ndarray] | None,
        f0_up_key: int,
        f0_method: PitchMethod,
        resample_sr: int,  # Target sample rate
        rms_mix_rate: float,
        protect: float,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[str, tuple[int, np.ndarray] | None]:
        if self.net_g is None or self.pipeline is None:
            return "Model not loaded. Please select a valid SID.", None
        if self.version not in {"v2", "v3"}:
            raise ValueError("Only v2/v3 models with f0 are supported.")
        f0_file = None
        sid = 0
        filter_radius = 3
        # protect safeguard
        try:
            protect = float(protect)
        except (TypeError, ValueError):
            logger.warning(f"Invalid protect value: {protect}. Defaulting to 0.33")
            protect = 0.33
        tgt_sr = self.tgt_sr
        if tgt_sr is None:
            return "Model target sample rate unknown. Please reload the model.", None
        # f0_up_key = f0_up_key
        try:
            if sr_and_audio is None:
                return "Audio is required", None

            original_sr, audio = sr_and_audio
            if original_sr != 16000:
                # print(f"Resampling audio from {original_sr} Hz to {16000} Hz")
                audio = resample_audio(audio, original_sr, 16000)
            audio_max: np.float64 = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = [0.0, 0.0, 0.0]
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config)
            assert self.hubert_model is not None

            audio_opt: np.ndarray = self.pipeline.pipeline(
                model=self.hubert_model,
                net_g=self.net_g,
                sid=sid,
                audio=audio,
                # input_audio_path="NA",
                times=times,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                # filter_radius=filter_radius,
                tgt_sr=tgt_sr,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                version=self.version,
                protect=protect,
                f0_file=f0_file,
                progress=progress,
            )
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            return (
                f"Success.\nTime:\nnpy: {times[0]:.2f}s, f0: {times[1]:.2f}s, infer: {times[2]:.2f}s.",
                (tgt_sr, audio_opt),
            )
        except:
            info = traceback.format_exc()
            logger.warning(info)
            return f"Failed with error:\n{info}", None
