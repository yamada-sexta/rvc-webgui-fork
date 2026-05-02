import sys
import traceback
from pathlib import Path

now_dir = Path.cwd()
sys.path.append(str(now_dir))

from tap import Tap
from loguru import logger
from lib.accelerate_utils import get_accelerator, use_half_precision


class ExtractFeatureCpuArgs(Tap):
    # Experiment directory.
    exp_dir: Path
    # Model version.
    version: str

    def configure(self) -> None:
        self.add_argument("exp_dir")
        self.add_argument("version")


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]) or len(sys.argv) == 3:
    parsed_args = ExtractFeatureCpuArgs().parse_args()
else:
    raise ValueError("Expected positional arguments: exp_dir version")
exp_dir = parsed_args.exp_dir
version = parsed_args.version
if version != "v2":
    raise ValueError("Only v2 feature extraction is supported.")
import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

accelerator = get_accelerator()
device = accelerator.device
is_half = use_half_precision()

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


logger.bind(event="feature_args", argv=sys.argv[1:]).info(
    "Received feature extraction args"
)
model_path = Path("assets/hubert/hubert_base.pt")

logger.info(f"Feature extraction output directory: {exp_dir}")
wavPath = exp_dir / "1_16k_wavs"
outPath = exp_dir / "3_feature768"
outPath.mkdir(parents=True, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
logger.bind(
    event="ui_progress",
    detail_event="feature_model_loading",
    stage="extract_feature",
    current=0,
    total=1,
    fraction=0.0,
    message="Loading HuBERT model...",
).info("Loading HuBERT model")
logger.info(f"Loading HuBERT model from {model_path}")
# if hubert model is exist
if not model_path.exists():
    logger.error(
        f"Feature extraction stopped because {model_path} does not exist. Download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
    )
    exit(0)

from fairseq.data.dictionary import Dictionary
from torch.serialization import safe_globals

with safe_globals([Dictionary]):
    # torch.serialization.add_safe_globals([Dictionary])
    models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(model_path)],
        suffix="",
    )
model = accelerator.prepare(models[0])
logger.bind(
    event="ui_progress",
    detail_event="feature_model_loaded",
    stage="extract_feature",
    current=0,
    total=1,
    fraction=0.0,
    message=f"HuBERT model loaded on {device}. Preparing feature extraction...",
).info("HuBERT model loaded")
logger.info(f"Moved HuBERT model to {device}")
if is_half:
    if device.type != "cpu":
        model = model.half()
model.eval()

todo = sorted(wavPath.iterdir(), key=lambda p: p.name)
if len(todo) == 0:
    logger.bind(event="feature_empty", total=0).info("No features to extract")
else:
    logger.bind(
        event="ui_progress",
        detail_event="feature_started",
        stage="extract_feature",
        current=0,
        total=len(todo),
        fraction=0.0,
        message=f"Starting feature extraction 0/{len(todo)}",
        version=version,
    ).info("Starting feature extraction")
    if saved_cfg is None:
        raise RuntimeError("HuBERT checkpoint did not include a saved config")
    normalize = saved_cfg.task.normalize
    for idx, file in enumerate(todo):
        try:
            if file.suffix == ".wav":
                wav_path = wavPath / file.name
                out_path = outPath / file.with_suffix(".npy").name
                logger.bind(
                    event="ui_progress",
                    detail_event="feature_processing",
                    stage="extract_feature",
                    current=idx,
                    total=len(todo),
                    fraction=idx / max(len(todo), 1),
                    message=f"Processing features {idx + 1}/{len(todo)}: {file.name}",
                    file=file.name,
                ).info(f"Starting feature extraction for {file.name}")

                skipped = out_path.exists()
                if not skipped:
                    feats = readwave(wav_path, normalize=normalize)
                    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                    inputs = {
                        "source": (
                            feats.half().to(device)
                            if is_half and device.type != "cpu"
                            else feats.to(device)
                        ),
                        "padding_mask": padding_mask.to(device),
                        "output_layer": 12,
                    }
                    with torch.no_grad():
                        logits = model.extract_features(**inputs)
                        feats = logits[0]

                    feats = feats.squeeze(0).float().cpu().numpy()
                    if np.isnan(feats).sum() == 0:
                        np.save(out_path, feats, allow_pickle=False)
                    else:
                        logger.warning(f"{file.name} contains NaN values")
                logger.bind(
                    event="ui_progress",
                    detail_event="feature_progress",
                    stage="extract_feature",
                    current=idx + 1,
                    total=len(todo),
                    fraction=(idx + 1) / max(len(todo), 1),
                    message=f"Extracting features {idx + 1}/{len(todo)}: {file.name}",
                    file=file.name,
                    skipped=skipped,
                ).info(f"Extracted features for {file.name}")
        except Exception:
            logger.bind(
                event="ui_progress",
                detail_event="feature_failed",
                stage="extract_feature",
                current=idx + 1,
                total=len(todo),
                fraction=(idx + 1) / max(len(todo), 1),
                message=f"Feature extraction failed at {idx + 1}/{len(todo)}: {file.name}",
                file=file.name,
                traceback=traceback.format_exc(),
            ).exception(f"Failed feature extraction for {file.name}")
    logger.bind(event="feature_finished", total=len(todo)).info(
        "Finished feature extraction"
    )
