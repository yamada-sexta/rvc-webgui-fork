import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file
from tap import Tap

now_dir = Path.cwd()
sys.path.append(str(now_dir))

from infer.lib.infer_pack.v3_models import FrozenDacCodec
from infer.lib.train.utils import load_wav_to_torch
from lib.accelerate_utils import get_accelerator


class ExtractDacArgs(Tap):
    # Experiment directory.
    exp_dir: Path
    # Source sample rate.
    sampling_rate: int
    # Hugging Face DAC model id.
    model_id: str = "descript/dac_16khz"

    def configure(self) -> None:
        self.add_argument("exp_dir")
        self.add_argument("sampling_rate")
        self.add_argument("--model_id")


args = ExtractDacArgs().parse_args()
exp_dir = args.exp_dir
accelerator = get_accelerator()

logger.remove()
logger.add(
    exp_dir / "extract_f0_feature.log",
    level="INFO",
    serialize=True,
    enqueue=False,
    backtrace=False,
    diagnose=False,
)
logger.add(
    sys.stderr,
    level="INFO",
    serialize=False,
    enqueue=False,
    backtrace=False,
    diagnose=False,
)

wav_dir = exp_dir / "0_gt_wavs"
out_dir = exp_dir / "4_dac"
out_dir.mkdir(parents=True, exist_ok=True)

logger.bind(
    event="ui_progress",
    detail_event="feature_model_loading",
    stage="extract_dac",
    current=0,
    total=1,
    fraction=0.0,
    message="Loading DAC model...",
).info("Loading DAC model")
codec = FrozenDacCodec(args.sampling_rate, args.model_id).to(accelerator.device)
codec.eval()

todo = sorted(wav_dir.glob("*.wav"), key=lambda p: p.name)
logger.bind(
    event="ui_progress",
    detail_event="feature_started",
    stage="extract_dac",
    current=0,
    total=len(todo),
    fraction=0.0,
    message=f"Starting DAC extraction 0/{len(todo)}",
).info("Starting DAC extraction")

for idx, wav_path in enumerate(todo):
    out_path = out_dir / f"{wav_path.stem}.safetensors"
    skipped = out_path.exists()
    logger.bind(
        event="ui_progress",
        detail_event="feature_processing",
        stage="extract_dac",
        current=idx,
        total=len(todo),
        fraction=idx / max(len(todo), 1),
        message=f"Processing DAC {idx + 1}/{len(todo)}: {wav_path.name}",
        file=wav_path.name,
    ).info(f"Starting DAC extraction for {wav_path.name}")
    if not skipped:
        audio, sampling_rate = load_wav_to_torch(wav_path)
        if sampling_rate != args.sampling_rate:
            raise ValueError(f"{wav_path} SR {sampling_rate} != {args.sampling_rate}")
        audio = audio.unsqueeze(0).unsqueeze(0).to(accelerator.device)
        with torch.no_grad():
            latents = codec.encode(audio).squeeze(0).float().cpu()
        save_file({"latents": latents}, out_path)
    logger.bind(
        event="ui_progress",
        detail_event="feature_progress",
        stage="extract_dac",
        current=idx + 1,
        total=len(todo),
        fraction=(idx + 1) / max(len(todo), 1),
        message=f"Extracting DAC {idx + 1}/{len(todo)}: {wav_path.name}",
        file=wav_path.name,
        skipped=skipped,
    ).info(f"Extracted DAC for {wav_path.name}")

logger.bind(event="feature_finished", total=len(todo)).info("Finished DAC extraction")
