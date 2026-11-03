import math

import numpy as np

_DERIVATIVE_CUTOFF_HERTZ = 1.0
_STRONGEST_MINIMUM_CUTOFF_HERTZ = 0.3
_WEAKEST_MINIMUM_CUTOFF_HERTZ = 3.0
_SPEED_COEFFICIENT = 2.0


def minimum_cutoff_for_smoothing(smoothing: float) -> float:
    clamped = min(max(smoothing, 0.0), 1.0)
    return _WEAKEST_MINIMUM_CUTOFF_HERTZ * (_STRONGEST_MINIMUM_CUTOFF_HERTZ / _WEAKEST_MINIMUM_CUTOFF_HERTZ) ** clamped


def smooth(values: np.ndarray, timestamps_seconds: np.ndarray, smoothing: float) -> np.ndarray:
    if smoothing <= 0.0 or len(values) < 2:
        return values.copy()
    minimum_cutoff = minimum_cutoff_for_smoothing(smoothing)
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    smoothed_derivative = np.zeros_like(values[0])
    for frame_index in range(1, len(values)):
        elapsed_seconds = max(float(timestamps_seconds[frame_index] - timestamps_seconds[frame_index - 1]), 1e-4)
        derivative = (values[frame_index] - smoothed[frame_index - 1]) / elapsed_seconds
        smoothed_derivative = smoothed_derivative + _alpha(_DERIVATIVE_CUTOFF_HERTZ, elapsed_seconds) * (
            derivative - smoothed_derivative
        )
        speed = np.linalg.norm(smoothed_derivative, axis=-1, keepdims=True)
        cutoff = minimum_cutoff + _SPEED_COEFFICIENT * speed
        alpha = _alpha(cutoff, elapsed_seconds)
        smoothed[frame_index] = smoothed[frame_index - 1] + alpha * (values[frame_index] - smoothed[frame_index - 1])
    return smoothed


def _alpha(cutoff_hertz: np.ndarray | float, elapsed_seconds: float) -> np.ndarray | float:
    time_constant = 1.0 / (2.0 * math.pi * cutoff_hertz)
    return 1.0 / (1.0 + time_constant / elapsed_seconds)
