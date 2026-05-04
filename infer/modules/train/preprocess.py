import sys
from pathlib import Path
from typing import Literal, cast

from scipy import signal
from tap import Tap

now_dir = Path.cwd()
sys.path.append(str(now_dir))
import traceback

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from loguru import logger

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer


class PreprocessArgs(Tap):
    # Input training audio directory.
    inp_root: Path
    # Target sample rate.
    sr: int
    # Number of preprocessing workers.
    n_p: int
    # Experiment output directory.
    exp_dir: Path
    # Run without multiprocessing.
    noparallel: bool = False
    # Maximum segment length in seconds.
    per: float


args = PreprocessArgs().parse_args()
inp_root = args.inp_root
sr = args.sr
n_p = args.n_p
exp_dir = args.exp_dir
noparallel = args.noparallel
per = args.per
logger.remove()
logger.add(
    exp_dir / "preprocess.log",
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


class PreProcess:
    slicer: Slicer
    sr: int
    bh: NDArray[np.floating]
    ah: NDArray[np.floating]
    per: float
    overlap: float
    tail: float
    max: float
    alpha: float
    exp_dir: Path
    gt_wavs_dir: Path
    wavs16k_dir: Path
    total_files: int

    def __init__(self: "PreProcess", sr: int, exp_dir: Path, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        bh, ah = cast(
            tuple[NDArray[np.floating], NDArray[np.floating]],
            signal.butter(N=5, Wn=48, btype="highpass", fs=self.sr, output="ba"),
        )
        self.bh = np.asarray(bh, dtype=np.float64)
        self.ah = np.asarray(ah, dtype=np.float64)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir: Path = exp_dir
        self.gt_wavs_dir: Path = exp_dir / "0_gt_wavs"
        self.wavs16k_dir: Path = exp_dir / "1_16k_wavs"
        self.total_files = 1
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.gt_wavs_dir.mkdir(parents=True, exist_ok=True)
        self.wavs16k_dir.mkdir(parents=True, exist_ok=True)

    def norm_write(
        self: "PreProcess", tmp_audio: np.ndarray, idx0: int, idx1: int
    ) -> None:
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            logger.warning(f"Skipping loud segment {idx0}_{idx1} with peak {tmp_max}")
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            self.gt_wavs_dir / f"{idx0}_{idx1}.wav",
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sr, target_sr=16000
        )  # , res_type="soxr_vhq"
        wavfile.write(
            self.wavs16k_dir / f"{idx0}_{idx1}.wav",
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self: "PreProcess", path: Path, idx0: int):
        try:
            audio = load_audio(path, self.sr)
            # zero phased digital filter cause pre-ringing noise...
            # audio = signal.filtfilt(self.bh, self.ah, audio)
            audio: NDArray[np.floating] = cast(
                NDArray[np.floating], signal.lfilter(self.bh, self.ah, audio)
            )

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while 1:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            logger.bind(
                event="ui_progress",
                detail_event="preprocess_file_done",
                current=idx0 + 1,
                total=self.total_files,
                fraction=(idx0 + 1) / max(self.total_files, 1),
                stage="preprocess",
                message=f"Preprocessing {idx0 + 1}/{self.total_files}: {path.name}",
                file=str(path),
            ).info(f"Preprocessed {path.name}")
        except Exception:
            logger.bind(
                event="ui_progress",
                detail_event="preprocess_file_failed",
                current=idx0 + 1,
                total=self.total_files,
                fraction=(idx0 + 1) / max(self.total_files, 1),
                stage="preprocess",
                message=f"Preprocess failed at {idx0 + 1}/{self.total_files}: {path.name}",
                file=str(path),
                traceback=traceback.format_exc(),
            ).exception(f"Failed to preprocess {path.name}")

    def pipeline_mp(self: "PreProcess", infos: list[tuple[Path, int]]) -> None:
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self: "PreProcess", inp_root: Path, n_p: int) -> None:
        try:
            infos = [
                (path, idx)
                for idx, path in enumerate(
                    sorted(inp_root.iterdir(), key=lambda p: p.name)
                )
            ]
            self.total_files = max(len(infos), 1)
            _ = n_p
            for path, idx0 in infos:
                self.pipeline(path, idx0)
        except Exception:
            logger.bind(
                event="preprocess_failed",
                traceback=traceback.format_exc(),
            ).exception("Preprocess stage failed")


def preprocess_trainset(inp_root: Path, sr: int, n_p: int, exp_dir: Path, per: float):
    pp = PreProcess(sr, exp_dir, per)
    logger.bind(
        event="preprocess_started",
        input_root=str(inp_root),
        sample_rate=sr,
        workers=n_p,
        noparallel=noparallel,
        segment_seconds=per,
    ).info("Starting preprocess")
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    logger.bind(event="preprocess_finished").info("Finished preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sr, n_p, exp_dir, per)
