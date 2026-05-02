import os
import sys
import traceback
from pathlib import Path

from tap import Tap
from loguru import logger

now_dir = os.getcwd()
sys.path.append(now_dir)

import numpy as np

from infer.lib.audio import load_audio
from lib.f0 import Generator, PitchMethod


class ExtractF0Args(Tap):
    # Experiment directory.
    exp_dir: Path
    # Number of CPU extraction workers.
    n_p: int
    # F0 extraction method.
    f0method: PitchMethod

    def configure(self) -> None:
        self.add_argument("exp_dir")
        self.add_argument("n_p")
        self.add_argument("f0method")


args = ExtractF0Args().parse_args()
exp_dir = args.exp_dir
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


n_p = args.n_p
f0method = args.f0method


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
            message="Loading F0 extractor...",
        ).info("Loading F0 extractor")
        self.f0_gen = Generator(
            Path("assets/rmvpe"),
            False,
            0,
            device="cpu",
            window=self.hop,
            sr=self.fs,
        )

    def go(self, paths, f0_method):
        if len(paths) == 0:
            logger.bind(event="f0_empty", total=0).info("No f0 files to process")
        else:
            logger.bind(
                event="ui_progress",
                detail_event="f0_started",
                stage="extract_f0",
                current=0,
                total=len(paths),
                fraction=0.0,
                message=f"Starting pitch extraction 0/{len(paths)}",
                method=f0_method,
            ).info("Starting f0 extraction")
            for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
                try:
                    logger.bind(
                        event="ui_progress",
                        detail_event="f0_processing",
                        stage="extract_f0",
                        current=idx,
                        total=len(paths),
                        fraction=idx / max(len(paths), 1),
                        message=f"Processing pitch {idx + 1}/{len(paths)}: {Path(inp_path).name}",
                        file=inp_path,
                    ).info(f"Starting f0 for {Path(inp_path).name}")
                    skipped = os.path.exists(opt_path1 + ".npy") and os.path.exists(
                        opt_path2 + ".npy"
                    )
                    if not skipped:
                        audio = load_audio(inp_path, self.fs)
                        p_len = audio.shape[0] // self.hop
                        coarse_pit, featur_pit = self.f0_gen.calculate(
                            audio,
                            p_len,
                            0,
                            f0_method,
                            3,
                        )
                        np.save(
                            opt_path2,
                            featur_pit,
                            allow_pickle=False,
                        )
                        np.save(
                            opt_path1,
                            coarse_pit,
                            allow_pickle=False,
                        )
                    logger.bind(
                        event="ui_progress",
                        detail_event="f0_progress",
                        stage="extract_f0",
                        current=idx + 1,
                        total=len(paths),
                        fraction=(idx + 1) / max(len(paths), 1),
                        message=f"Extracting pitch {idx + 1}/{len(paths)}: {Path(inp_path).name}",
                        file=inp_path,
                        skipped=skipped,
                    ).info(f"Processed f0 for {Path(inp_path).name}")
                except Exception:
                    logger.bind(
                        event="ui_progress",
                        detail_event="f0_failed",
                        stage="extract_f0",
                        current=idx + 1,
                        total=len(paths),
                        fraction=(idx + 1) / max(len(paths), 1),
                        message=f"Pitch extraction failed at {idx + 1}/{len(paths)}: {Path(inp_path).name}",
                        file=inp_path,
                        traceback=traceback.format_exc(),
                    ).exception(f"Failed f0 extraction for {Path(inp_path).name}")


if __name__ == "__main__":
    logger.bind(event="f0_args", argv=sys.argv[1:]).info("Received f0 extraction args")
    featureInput = FeatureInput()
    paths = []
    inp_root = exp_dir / "1_16k_wavs"
    opt_root1 = exp_dir / "2a_f0"
    opt_root2 = exp_dir / "2b-f0nsf"

    opt_root1.mkdir(parents=True, exist_ok=True)
    opt_root2.mkdir(parents=True, exist_ok=True)
    for wav_file in sorted(inp_root.iterdir(), key=lambda p: p.name):
        inp_path = inp_root / wav_file.name
        if "spec" in inp_path.name:
            continue
        opt_path1 = opt_root1 / wav_file.name
        opt_path2 = opt_root2 / wav_file.name
        paths.append([str(inp_path), str(opt_path1), str(opt_path2)])
    featureInput.go(paths, f0method)
