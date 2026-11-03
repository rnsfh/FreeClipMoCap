from pathlib import Path

import aud
import numpy as np


class VideoAudioError(Exception):
    pass


def read_audio(video_path: Path) -> tuple[np.ndarray, float]:
    try:
        sound = aud.Sound(str(video_path))
        sample_rate, _channel_count = sound.specs
        samples = np.asarray(sound.data(), dtype=np.float32)
    except Exception as error:
        raise VideoAudioError(f"Cannot read the sound of {video_path.name}: {error}") from error
    if samples.size == 0:
        raise VideoAudioError(f"{video_path.name} has no sound.")
    return samples, float(sample_rate)
