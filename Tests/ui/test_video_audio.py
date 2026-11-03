import tempfile
import unittest
from pathlib import Path

import aud
import numpy as np

from bl_ext.user_default.freeclipmocap.ui.video_audio import VideoAudioError, read_audio


class VideoAudioTests(unittest.TestCase):
    def test_reads_the_sound_and_its_rate(self) -> None:
        samples = (0.1 * np.sin(np.arange(44100) / 44100.0 * 2.0 * np.pi * 440.0)).astype(np.float32).reshape(-1, 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tone.wav"
            aud.Sound.buffer(samples, 44100.0).write(
                str(path), rate=44100, channels=aud.CHANNELS_MONO, format=aud.FORMAT_FLOAT32,
                container=aud.CONTAINER_WAV, codec=aud.CODEC_PCM,
            )

            read_samples, sample_rate = read_audio(path)

        self.assertEqual(sample_rate, 44100.0)
        np.testing.assert_allclose(read_samples[: len(samples)], samples, atol=1e-6)

    def test_a_file_without_sound_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "not sound.mp4"
            path.write_bytes(b"not a video")
            with self.assertRaises(VideoAudioError):
                read_audio(path)


if __name__ == "__main__":
    unittest.main()
