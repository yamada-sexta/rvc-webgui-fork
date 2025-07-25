import platform, os
import ffmpeg
import numpy as np
import av
from io import BytesIO
import traceback
import re


def wav2(input_path: str, output_path: str, format: str):
    inp = av.open(input_path, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(output_path, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def load_audio(file: str, sr: int) -> np.ndarray:
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) == False:
            raise RuntimeError(f"Wrong audio path: {file}")
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str: str) -> str:
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(
        r"[\u202a\u202b\u202c\u202d\u202e]", "", path_str
    )  # 移除 Unicode 控制字符
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
