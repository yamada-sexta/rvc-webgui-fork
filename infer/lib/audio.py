import platform
import numpy as np
import av
import av.audio.resampler
import re
from av.audio.frame import AudioFrame
from av.audio.stream import AudioStream
from numpy.typing import NDArray
from pathlib import Path


def load_audio(file: Path, sr: int) -> NDArray[np.float32]:
    try:
        with av.open(file, "r") as container:
            stream = next(s for s in container.streams if s.type == "audio")

            resampler = av.audio.resampler.AudioResampler(
                format="flt", layout="mono", rate=sr
            )

            audio_data: list[NDArray[np.float32]] = []
            for frame in container.decode(stream):
                if not isinstance(frame, AudioFrame):
                    continue
                # Resample returns either a frame or a list of frames
                resampled = resampler.resample(frame)
                if not resampled:
                    continue
                if isinstance(resampled, list):
                    frames = resampled
                else:
                    frames = [resampled]

                for f in frames:
                    arr = np.asarray(f.to_ndarray(), dtype=np.float32)
                    audio_data.append(arr)

            return np.concatenate(audio_data, axis=1).flatten()
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with PyAV: {e}") from e
