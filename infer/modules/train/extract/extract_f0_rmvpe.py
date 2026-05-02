import os
import sys
import traceback
from pathlib import Path

from tap import Tap
from loguru import logger

now_dir = Path.cwd()
sys.path.append(str(now_dir))

import numpy as np

from infer.lib.audio import load_audio
from lib.f0 import Generator
from lib.accelerate_utils import device_string, get_accelerator, use_half_precision


class ExtractF0RmvpeArgs(Tap):
    # Experiment directory.
    exp_dir: Path

    def configure(self) -> None:
        self.add_argument("exp_dir")


args = ExtractF0RmvpeArgs().parse_args()
exp_dir = args.exp_dir
accelerator = get_accelerator()
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


class FeatureInput:
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        logger.bind(
            event="ui_progress",
            detail_event="f0_model_loading",
            stage="extract_f0",
            current=0,
            total=1,
            fraction=0.0,
            message="Loading RMVPE model...",
            device=device_string(),
        ).info("Loading RMVPE model")
        self.f0_gen = Generator(
            Path("assets/rmvpe"),
            is_half,
            0,
            device=device_string(),
            window=self.hop,
            sr=self.fs,
        )

    def go(self, paths, f0_method):
        if len(paths) == 0:
            logger.bind(event="f0_empty", total=0, method=f0_method).info(
                "No RMVPE f0 files to process"
            )
        else:
            logger.bind(
                event="ui_progress",
                detail_event="f0_started",
                stage="extract_f0",
                current=0,
                total=len(paths),
                fraction=0.0,
                message=f"Starting RMVPE pitch extraction 0/{len(paths)}",
                method=f0_method,
                device=device_string(),
            ).info("Starting RMVPE f0 extraction")
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    logger.bind(
                        event="ui_progress",
                        detail_event="f0_processing",
                        stage="extract_f0",
                        current=idx,
                        total=len(paths),
                        fraction=idx / max(len(paths), 1),
                        message=f"Processing RMVPE pitch {idx + 1}/{len(paths)}: {Path(inp_path).name}",
                        file=inp_path,
                        device=device_string(),
                    ).info(f"Starting RMVPE f0 for {Path(inp_path).name}")
                    skipped = (
                        Path(f"{opt_path1}.npy").exists()
                        and Path(f"{opt_path2}.npy").exists()
                    )
                    if not skipped:
                        audio = load_audio(inp_path, self.fs)
                        p_len = audio.shape[0] // self.hop
                        coarse_pit, featur_pit = self.f0_gen.calculate(
                            audio,
                            p_len,
                            0,
                            "rmvpe",
                            3,
                        )
                        np.save(opt_path2, featur_pit, allow_pickle=False)
                        np.save(opt_path1, coarse_pit, allow_pickle=False)
                    logger.bind(
                        event="ui_progress",
                        detail_event="f0_progress",
                        stage="extract_f0",
                        current=idx + 1,
                        total=len(paths),
                        fraction=(idx + 1) / max(len(paths), 1),
                        message=f"Extracting RMVPE pitch {idx + 1}/{len(paths)}: {Path(inp_path).name}",
                        file=inp_path,
                        skipped=skipped,
                        device=device_string(),
                    ).info(f"Processed RMVPE f0 for {Path(inp_path).name}")
                except Exception:
                    logger.bind(
                        event="ui_progress",
                        detail_event="f0_failed",
                        stage="extract_f0",
                        current=idx + 1,
                        total=len(paths),
                        fraction=(idx + 1) / max(len(paths), 1),
                        message=f"RMVPE pitch extraction failed at {idx + 1}/{len(paths)}: {Path(inp_path).name}",
                        file=inp_path,
                        device=device_string(),
                        traceback=traceback.format_exc(),
                    ).exception(f"Failed RMVPE f0 extraction for {Path(inp_path).name}")


if __name__ == "__main__":
    logger.bind(event="f0_args", argv=sys.argv[1:], device=device_string()).info(
        "Received RMVPE extraction args"
    )
    featureInput = FeatureInput()
    paths = []
    inp_root = exp_dir / "1_16k_wavs"
    opt_root1 = exp_dir / "2a_f0"
    opt_root2 = exp_dir / "2b-f0nsf"

    opt_root1.mkdir(parents=True, exist_ok=True)
    opt_root2.mkdir(parents=True, exist_ok=True)
    for wav_file in sorted(inp_root.iterdir(), key=lambda path: path.name):
        inp_path = wav_file
        if "spec" in inp_path.name:
            continue
        opt_path1 = opt_root1 / wav_file.name
        opt_path2 = opt_root2 / wav_file.name
        paths.append([str(inp_path), str(opt_path1), str(opt_path2)])
    try:
        featureInput.go(paths, "rmvpe")
    except Exception:
        logger.bind(
            event="f0_failed",
            traceback=traceback.format_exc(),
            device=device_string(),
        ).exception("RMVPE extraction stage failed")
