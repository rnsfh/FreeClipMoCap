import unittest

import numpy as np

from bl_ext.user_default.freeclipmocap.solver.audio_sync import AudioSyncError, find_start_delay


def room_sound(duration_seconds: float, sample_rate: float, event_times_seconds: list[float], seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    times = np.arange(int(duration_seconds * sample_rate)) / sample_rate
    sound = 0.003 * generator.normal(size=len(times))
    for event_time in event_times_seconds:
        after = times - event_time
        burst = after >= 0.0
        sound[burst] += 0.6 * np.exp(-after[burst] / 0.015) * np.sin(2.0 * np.pi * 1800.0 * after[burst])
    return sound


class AudioSyncTests(unittest.TestCase):
    _EVENTS = [1.2, 2.05, 3.9, 4.15, 6.6, 7.3]

    def test_finds_a_later_start_across_sample_rates_and_gains(self) -> None:
        reference = room_sound(9.0, 48000.0, self._EVENTS, seed=1)
        later_by = 0.3712
        other = 0.4 * room_sound(8.0, 44100.0, [time - later_by for time in self._EVENTS], seed=2)

        sync = find_start_delay(np.column_stack((reference, reference)), 48000.0, other, 44100.0, largest_delay_seconds=5.0)

        self.assertAlmostEqual(sync.other_started_later_seconds, later_by, delta=0.002)
        self.assertGreater(sync.peak_ratio, 1.5)

    def test_finds_an_earlier_start(self) -> None:
        earlier_by = 1.1047
        reference = room_sound(8.0, 48000.0, self._EVENTS, seed=3)
        other = room_sound(10.0, 48000.0, [time + earlier_by for time in self._EVENTS], seed=4)

        sync = find_start_delay(reference, 48000.0, other, 48000.0, largest_delay_seconds=5.0)

        self.assertAlmostEqual(sync.other_started_later_seconds, -earlier_by, delta=0.002)

    def test_finds_one_clap_in_a_quiet_take_with_changing_hiss(self) -> None:
        sample_rate = 48000.0
        later_by = 0.3347
        clap_time = 2.6

        def quiet_take(duration_seconds: float, clap_at_seconds: float, seed: int) -> np.ndarray:
            generator = np.random.default_rng(seed)
            sample_count = int(duration_seconds * sample_rate)
            stretch_count = int(duration_seconds * 4.0) + 1
            hiss_level = 10.0 ** (generator.uniform(-90.0, -60.0, stretch_count) / 20.0)
            stretch = np.minimum((np.arange(sample_count) / sample_rate * 4.0).astype(np.int64), stretch_count - 1)
            sound = hiss_level[stretch] * generator.normal(size=sample_count)
            after_clap = np.arange(sample_count) / sample_rate - clap_at_seconds
            clap = after_clap >= 0.0
            sound[clap] += 0.3 * np.exp(-after_clap[clap] / 0.015) * np.sin(2.0 * np.pi * 1800.0 * after_clap[clap])
            return sound

        reference = quiet_take(35.0, clap_time, seed=6)
        other = quiet_take(36.0, clap_time - later_by, seed=8)

        sync = find_start_delay(reference, sample_rate, other, sample_rate, largest_delay_seconds=30.0)

        self.assertAlmostEqual(sync.other_started_later_seconds, later_by, delta=0.002)
        self.assertGreater(sync.peak_ratio, 1.3)

    def test_silence_is_refused(self) -> None:
        with self.assertRaises(AudioSyncError):
            find_start_delay(np.zeros(48000), 48000.0, room_sound(1.0, 48000.0, [0.5], seed=5), 48000.0, 1.0)


if __name__ == "__main__":
    unittest.main()
