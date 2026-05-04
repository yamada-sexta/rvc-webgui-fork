import sys
from pathlib import Path

now_dir = Path.cwd()
sys.path.append(str(now_dir))
from dotenv import load_dotenv
from scipy.io import wavfile
from tap import Tap

# from lib.f0 import PitchMethod
from lib.f0.type import PitchMethod

####
# USAGE
#
# In your Terminal or CMD or whatever


class InferArgs(Tap):

    f0up_key: int = 0  # Pitch shift in semitones.

    input_path: Path  # Input audio path.

    f0method: PitchMethod = "harvest"  # F0 extraction method.

    opt_path: Path  # Output audio path.

    model_name: str  # Model name stored in assets/weights.

    filter_radius: int = 3  # Median filter radius for extracted pitch.

    resample_sr: int = 0  # Resample output sample rate, or 0 to keep model rate.

    rms_mix_rate: float = 1  # RMS envelope mix rate.

    protect: float = 0.33  # Protect unvoiced consonants.


def arg_parse() -> InferArgs:
    args = InferArgs().parse_args()
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
    audio = load_audio(input_path, 16000)
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
    wavfile.write(output_path, wav_opt[0], wav_opt[1])


if __name__ == "__main__":
    main()
