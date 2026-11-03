from collections.abc import Callable

import numpy as np

from ..solver import rotations


def angle_between_rotations(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(rotations.normalized(first) * rotations.normalized(second), axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def select_key_frames(quaternions: np.ndarray, tolerance_degrees: float) -> np.ndarray:
    def errors(start: int, end: int, inner: np.ndarray, fractions: np.ndarray) -> np.ndarray:
        interpolated = rotations.slerp(quaternions[start], quaternions[end], fractions)
        return angle_between_rotations(interpolated, quaternions[inner])

    return _douglas_peucker(len(quaternions), errors, np.radians(tolerance_degrees))


def select_position_key_frames(positions: np.ndarray, tolerance: float) -> np.ndarray:
    def errors(start: int, end: int, inner: np.ndarray, fractions: np.ndarray) -> np.ndarray:
        interpolated = positions[start] + fractions[:, None] * (positions[end] - positions[start])
        return np.linalg.norm(interpolated - positions[inner], axis=-1)

    return _douglas_peucker(len(positions), errors, tolerance)


def _douglas_peucker(
    frame_count: int, errors_between: Callable[[int, int, np.ndarray, np.ndarray], np.ndarray], tolerance: float
) -> np.ndarray:
    if frame_count <= 2:
        return np.arange(frame_count)
    kept = {0, frame_count - 1}
    segments = [(0, frame_count - 1)]
    while segments:
        start, end = segments.pop()
        if end - start < 2:
            continue
        inner = np.arange(start + 1, end)
        errors = errors_between(start, end, inner, (inner - start) / (end - start))
        worst = int(np.argmax(errors))
        if errors[worst] > tolerance:
            middle = int(inner[worst])
            kept.add(middle)
            segments += [(start, middle), (middle, end)]
    return np.array(sorted(kept))


def worst_frames_of_each_run(errors: np.ndarray, tolerance_radians: float) -> np.ndarray:
    is_bad = errors > tolerance_radians
    worst_frames: list[int] = []
    frame_index = 0
    while frame_index < len(errors):
        if not is_bad[frame_index]:
            frame_index += 1
            continue
        run_end = frame_index
        while run_end < len(errors) and is_bad[run_end]:
            run_end += 1
        worst_frames.append(frame_index + int(np.argmax(errors[frame_index:run_end])))
        frame_index = run_end
    return np.array(worst_frames, dtype=int)
