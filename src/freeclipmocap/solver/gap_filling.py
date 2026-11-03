import numpy as np


def hold_last_good(values: np.ndarray, found: np.ndarray) -> np.ndarray | None:
    found_indices = np.flatnonzero(found)
    if len(found_indices) == 0:
        return None
    source_indices = np.maximum.accumulate(np.where(found, np.arange(len(found)), -1))
    source_indices[source_indices < 0] = found_indices[0]
    return values[source_indices]


def hold_last_good_per_point(values: np.ndarray, found: np.ndarray) -> np.ndarray:
    frame_count, point_count = found.shape
    frame_indices = np.broadcast_to(np.arange(frame_count)[:, None], found.shape)
    source_indices = np.maximum.accumulate(np.where(found, frame_indices, -1), axis=0)
    first_found = np.argmax(found, axis=0)
    source_indices = np.where(source_indices < 0, first_found[None, :], source_indices)
    filled = values[source_indices, np.arange(point_count)[None, :]]
    filled[:, ~found.any(axis=0)] = np.nan
    return filled


def blend_gaps(
    values: np.ndarray,
    found: np.ndarray,
    timestamps_seconds: np.ndarray,
    longest_blend_seconds: float,
    ease_seconds: float,
) -> np.ndarray | None:
    filled = hold_last_good(values, found)
    if filled is None:
        return None
    found_indices = np.flatnonzero(found)
    for gap_index in np.flatnonzero(np.diff(found_indices) > 1):
        before, after = int(found_indices[gap_index]), int(found_indices[gap_index + 1])
        start, end = float(timestamps_seconds[before]), float(timestamps_seconds[after])
        if end - start > longest_blend_seconds:
            start = end - ease_seconds
        fractions = np.clip((timestamps_seconds[before + 1:after] - start) / max(end - start, 1e-9), 0.0, 1.0)
        eased = (fractions * fractions * (3.0 - 2.0 * fractions)).reshape((-1,) + (1,) * (values.ndim - 1))
        filled[before + 1:after] = values[before] + eased * (values[after] - values[before])
    return filled
