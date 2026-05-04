import os
import sys
from pathlib import Path

print("Command-line arguments:", sys.argv)

now_dir = Path.cwd()
sys.path.append(str(now_dir))

import tqdm as tq
from dotenv import load_dotenv
from scipy.io import wavfile
from tap import Tap

from lib.f0.type import PitchMethod


class InferBatchArgs(Tap):
    # Pitch shift in semitones.
    f0up_key: int = 0
    # Input directory containing wav files.
    input_path: str
    # F0 extraction method.
    f0method: PitchMethod = "harvest"
    # Output directory.
    opt_path: str
    # Model name stored in assets/weights.
    model_name: str
    # Median filter radius for extracted pitch.
    filter_radius: int = 3
    # Resample output sample rate, or 0 to keep model rate.
    resample_sr: int = 0
    # RMS envelope mix rate.
    rms_mix_rate: float = 1
    # Protect unvoiced consonants.
    protect: float = 0.33


def arg_parse() -> InferBatchArgs:
    args = InferBatchArgs().parse_args()
    sys.argv = sys.argv[:1]
    return args


def main() -> None:
    load_dotenv()
    args = arg_parse()
    from configs.config import get_config
    from infer.lib.audio import load_audio
    from infer.modules.vc.modules import VC

    config = get_config()
    vc = VC(config)
    vc.get_vc(args.model_name)
    input_path = Path(args.input_path)
    output_path = Path(args.opt_path)
    for file_path in tq.tqdm(sorted(input_path.iterdir())):
        if file_path.suffix == ".wav":
            audio = load_audio(file_path, 16000)
            message, wav_opt = vc.vc_single(
                (16000, audio),
                args.f0up_key,
                args.f0method,
                args.resample_sr,
                args.rms_mix_rate,
                args.protect,
            )
            if wav_opt is None:
                raise RuntimeError(message)
            wavfile.write(output_path / file_path.name, wav_opt[0], wav_opt[1])


if __name__ == "__main__":
    main()
