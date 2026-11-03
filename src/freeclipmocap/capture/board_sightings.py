import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class BoardSightings:
    video_file_name: str
    image_width: int
    image_height: int
    frame_rate: float
    frame_indices: np.ndarray
    timestamps_seconds: np.ndarray
    corner_points: np.ndarray

    @property
    def corners_seen(self) -> np.ndarray:
        return np.isfinite(self.corner_points[:, :, 0]).sum(axis=1)

    def corner_speeds_pixels_per_second(self) -> np.ndarray:
        speeds = np.zeros(len(self.frame_indices))
        if len(speeds) < 2:
            return speeds
        steps = np.linalg.norm(np.diff(self.corner_points, axis=0), axis=2)
        durations = np.maximum(np.diff(self.timestamps_seconds), 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            step_speeds = np.nanmedian(steps, axis=1) / durations
        step_speeds = np.where(np.isfinite(step_speeds), step_speeds, 0.0)
        speeds[:-1] = step_speeds
        speeds[1:] = np.maximum(speeds[1:], step_speeds)
        return speeds

    def interpolated_at(self, times_seconds: np.ndarray) -> np.ndarray:
        later = np.searchsorted(self.timestamps_seconds, times_seconds, side="right")
        earlier = later - 1
        inside = (earlier >= 0) & (later < len(self.timestamps_seconds))
        earlier = np.clip(earlier, 0, len(self.timestamps_seconds) - 1)
        later = np.clip(later, 0, len(self.timestamps_seconds) - 1)
        consecutive = inside & (self.frame_indices[later] - self.frame_indices[earlier] == 1)
        spans = self.timestamps_seconds[later] - self.timestamps_seconds[earlier]
        weights = np.where(spans > 0.0, (times_seconds - self.timestamps_seconds[earlier]) / np.where(spans > 0.0, spans, 1.0), 0.0)
        points = (1.0 - weights)[:, None, None] * self.corner_points[earlier] + weights[:, None, None] * self.corner_points[later]
        points[~consecutive] = np.nan
        return points
