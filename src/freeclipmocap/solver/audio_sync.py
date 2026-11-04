from dataclasses import dataclass

import numpy as np

_ENVELOPE_RATE = 1000.0
_SMALLEST_ENERGY = 1e-10
_NOISE_FLOOR_SHARE_OF_MEDIAN = 10.0
_PEAK_EXCLUSION_SECONDS = 0.02


class AudioSyncError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class AudioSync:
    other_started_later_seconds: float
    peak_ratio: float


def onset_envelope(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    changes = np.diff(mono.astype(np.float64))
    hop_count = int(len(changes) * _ENVELOPE_RATE / sample_rate)
    if hop_count < 2 or sample_rate < _ENVELOPE_RATE:
        return np.zeros(0)
    boundaries = np.round(np.arange(hop_count + 1) * (sample_rate / _ENVELOPE_RATE)).astype(np.int64)
    summed_squares = np.concatenate(([0.0], np.cumsum(np.square(changes))))
    energy = np.diff(summed_squares[boundaries]) / np.diff(boundaries)
    # A floor well above the take's typical (median) loudness, so hiss and its level changes in a quiet room make
    # no onsets; only sounds that rise clearly above it, such as a clap, are matched.
    noise_floor = max(float(np.median(energy)) * _NOISE_FLOOR_SHARE_OF_MEDIAN, _SMALLEST_ENERGY)
    loudness = np.log(energy + noise_floor)
    onsets = np.maximum(np.diff(loudness, prepend=loudness[0]), 0.0)
    spread = onsets.std()
    return (onsets - onsets.mean()) / spread if spread > 0.0 else np.zeros_like(onsets)


def find_start_delay(
    reference_samples: np.ndarray,
    reference_rate: float,
    other_samples: np.ndarray,
    other_rate: float,
    largest_delay_seconds: float,
) -> AudioSync:
    reference = onset_envelope(reference_samples, reference_rate)
    other = onset_envelope(other_samples, other_rate)
    if not reference.any() or not other.any():
        raise AudioSyncError("One of the videos has no usable sound (silent or too short).")
    largest_lag = int(largest_delay_seconds * _ENVELOPE_RATE)
    length = 1 << int(np.ceil(np.log2(len(reference) + len(other))))
    # correlation[lag] = sum over n of reference[n] * other[n - lag]; negative lags wrap to the end.
    correlation = np.fft.irfft(np.fft.rfft(reference, length) * np.conj(np.fft.rfft(other, length)), length)
    lags = np.arange(-min(largest_lag, len(other) - 1), min(largest_lag, len(reference) - 1) + 1)
    values = correlation[lags % length]
    best = int(np.argmax(values))
    exclusion = int(_PEAK_EXCLUSION_SECONDS * _ENVELOPE_RATE)
    others = np.concatenate((values[: max(best - exclusion, 0)], values[best + exclusion + 1 :]))
    runner_up = float(others.max()) if len(others) else 0.0
    peak_ratio = float(values[best]) / runner_up if runner_up > 0.0 else float("inf")
    return AudioSync(
        other_started_later_seconds=(lags[best] + _parabolic_offset(values, best)) / _ENVELOPE_RATE,
        peak_ratio=peak_ratio,
    )


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    if index == 0 or index == len(values) - 1:
        return 0.0
    left, middle, right = values[index - 1], values[index], values[index + 1]
    curvature = left - 2.0 * middle + right
    return 0.5 * (left - right) / curvature if curvature < 0.0 else 0.0
