from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from .capture_settings import CaptureSettings

_FORMAT_VERSION = 2
_FOLDER_NAME = "FreeClipMocap"
_FILE_SUFFIX = ".landmarks.npz"


class CacheFormatError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class VideoSource:
    file_size: int
    modified_time_nanoseconds: int
    frame_rate: float
    width: int
    height: int

    def is_same_file(self, video_path: Path) -> bool:
        status = video_path.stat()
        return status.st_size == self.file_size and status.st_mtime_ns == self.modified_time_nanoseconds


@dataclass(frozen=True, slots=True, eq=False)
class PoseTrack:
    found: np.ndarray
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class HandTrack:
    found: np.ndarray
    image_landmarks: np.ndarray
    world_landmarks: np.ndarray
    score: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class LandmarkCache:
    tracking_model: str
    settings: CaptureSettings
    video: VideoSource
    frame_indices: np.ndarray
    timestamps_milliseconds: np.ndarray
    pose: PoseTrack
    left_hand: HandTrack
    right_hand: HandTrack

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    def sliced(self, start_index: int, stop_index: int) -> "LandmarkCache":
        frames = slice(start_index, stop_index)
        return replace(
            self,
            frame_indices=self.frame_indices[frames],
            timestamps_milliseconds=self.timestamps_milliseconds[frames],
            pose=_sliced_track(self.pose, frames),
            left_hand=_sliced_track(self.left_hand, frames),
            right_hand=_sliced_track(self.right_hand, frames),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(path.name + ".writing")
        with open(temporary_path, "wb") as cache_file:
            np.savez(cache_file, **self._arrays_by_name())
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LandmarkCache":
        with np.load(path, allow_pickle=False) as arrays:
            format_version = int(arrays["format_version"])
            if format_version != _FORMAT_VERSION:
                raise CacheFormatError(f"{path} has format version {format_version}, expected {_FORMAT_VERSION}.")
            return cls(
                tracking_model=str(arrays["tracking_model"]),
                settings=CaptureSettings.from_json(str(arrays["settings_json"])),
                video=VideoSource(
                    file_size=int(arrays["video_file_size"]),
                    modified_time_nanoseconds=int(arrays["video_modified_time_nanoseconds"]),
                    frame_rate=float(arrays["video_frame_rate"]),
                    width=int(arrays["video_width"]),
                    height=int(arrays["video_height"]),
                ),
                frame_indices=_read_only(arrays["frame_indices"]),
                timestamps_milliseconds=_read_only(arrays["timestamps_milliseconds"]),
                pose=PoseTrack(
                    found=_read_only(arrays["pose_found"]),
                    image_landmarks=_read_only(arrays["pose_image_landmarks"]),
                    world_landmarks=_read_only(arrays["pose_world_landmarks"]),
                    visibility=_read_only(arrays["pose_visibility"]),
                    presence=_read_only(arrays["pose_presence"]),
                ),
                left_hand=_load_hand_track(arrays, "left_hand"),
                right_hand=_load_hand_track(arrays, "right_hand"),
            )

    def _arrays_by_name(self) -> dict[str, np.ndarray]:
        return {
            "format_version": np.array(_FORMAT_VERSION, dtype=np.int32),
            "tracking_model": np.array(self.tracking_model),
            "settings_json": np.array(self.settings.to_json()),
            "video_file_size": np.array(self.video.file_size, dtype=np.int64),
            "video_modified_time_nanoseconds": np.array(self.video.modified_time_nanoseconds, dtype=np.int64),
            "video_frame_rate": np.array(self.video.frame_rate, dtype=np.float64),
            "video_width": np.array(self.video.width, dtype=np.int32),
            "video_height": np.array(self.video.height, dtype=np.int32),
            "frame_indices": self.frame_indices,
            "timestamps_milliseconds": self.timestamps_milliseconds,
            "pose_found": self.pose.found,
            "pose_image_landmarks": self.pose.image_landmarks,
            "pose_world_landmarks": self.pose.world_landmarks,
            "pose_visibility": self.pose.visibility,
            "pose_presence": self.pose.presence,
            **_hand_track_arrays_by_name(self.left_hand, "left_hand"),
            **_hand_track_arrays_by_name(self.right_hand, "right_hand"),
        }


def cache_path_for(video_path: Path, settings: CaptureSettings) -> Path:
    file_name = f"{video_path.name}.{settings.pose_model_variant}-{settings.short_hash()}{_FILE_SUFFIX}"
    return video_path.parent / _FOLDER_NAME / file_name


def find_current_cache(video_path: Path, settings: CaptureSettings) -> Path | None:
    path = cache_path_for(video_path, settings)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            is_current = (
                int(arrays["format_version"]) == _FORMAT_VERSION
                and str(arrays["settings_json"]) == settings.to_json()
                and int(arrays["video_file_size"]) == video_path.stat().st_size
                and int(arrays["video_modified_time_nanoseconds"]) == video_path.stat().st_mtime_ns
            )
    except (OSError, ValueError, KeyError):
        return None
    return path if is_current else None


def _hand_track_arrays_by_name(track: HandTrack, prefix: str) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_found": track.found,
        f"{prefix}_image_landmarks": track.image_landmarks,
        f"{prefix}_world_landmarks": track.world_landmarks,
        f"{prefix}_score": track.score,
    }


def _load_hand_track(arrays: np.lib.npyio.NpzFile, prefix: str) -> HandTrack:
    return HandTrack(
        found=_read_only(arrays[f"{prefix}_found"]),
        image_landmarks=_read_only(arrays[f"{prefix}_image_landmarks"]),
        world_landmarks=_read_only(arrays[f"{prefix}_world_landmarks"]),
        score=_read_only(arrays[f"{prefix}_score"]),
    )


def _sliced_track[Track: (PoseTrack, HandTrack)](track: Track, frames: slice) -> Track:
    return replace(track, **{field.name: getattr(track, field.name)[frames] for field in fields(track)})


def _read_only(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array
