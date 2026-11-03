import contextlib
import threading
from pathlib import Path
from typing import Any

import cv2
import mediapipe
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ..constants import hand_landmarks, pose_landmarks, tracking_models
from .capture_progress import CaptureCancelled, CaptureProgress
from .capture_settings import CaptureSettings
from .hand_crops import CropBox
from .landmark_cache import HandTrack, LandmarkCache, PoseTrack, VideoSource

_HAND_MODEL_FILE_NAME = "hand_landmarker.task"
_FALLBACK_FRAME_RATE = 30.0
_SMALLEST_WRIST_VISIBILITY = 0.3
_FARTHEST_WRIST_SHARE_OF_CROP = 0.12
_HAND_POINTS = {
    "left": (pose_landmarks.LEFT_WRIST, pose_landmarks.LEFT_INDEX, pose_landmarks.LEFT_PINKY),
    "right": (pose_landmarks.RIGHT_WRIST, pose_landmarks.RIGHT_INDEX, pose_landmarks.RIGHT_PINKY),
}


class CaptureError(Exception):
    pass


class _PoseRecorder:
    def __init__(self) -> None:
        self._found: list[bool] = []
        self._image_landmarks: list[np.ndarray] = []
        self._world_landmarks: list[np.ndarray] = []
        self._visibility: list[np.ndarray] = []
        self._presence: list[np.ndarray] = []

    def add(self, image_landmarks: list[Any], world_landmarks: list[Any]) -> None:
        self._found.append(True)
        self._image_landmarks.append(_positions(image_landmarks))
        self._world_landmarks.append(_positions(world_landmarks))
        self._visibility.append(_optional_values(landmark.visibility for landmark in image_landmarks))
        self._presence.append(_optional_values(landmark.presence for landmark in image_landmarks))

    def add_missing(self) -> None:
        self._found.append(False)
        self._image_landmarks.append(_missing_positions(pose_landmarks.LANDMARK_COUNT))
        self._world_landmarks.append(_missing_positions(pose_landmarks.LANDMARK_COUNT))
        self._visibility.append(np.full(pose_landmarks.LANDMARK_COUNT, np.nan, dtype=np.float32))
        self._presence.append(np.full(pose_landmarks.LANDMARK_COUNT, np.nan, dtype=np.float32))

    def finish(self) -> PoseTrack:
        return PoseTrack(
            found=np.array(self._found, dtype=bool),
            image_landmarks=np.stack(self._image_landmarks),
            world_landmarks=np.stack(self._world_landmarks),
            visibility=np.stack(self._visibility),
            presence=np.stack(self._presence),
        )


class _HandRecorder:
    def __init__(self) -> None:
        self._found: list[bool] = []
        self._image_landmarks: list[np.ndarray] = []
        self._world_landmarks: list[np.ndarray] = []
        self._score: list[float] = []

    def add(self, image_landmarks: np.ndarray, world_landmarks: np.ndarray, score: float) -> None:
        self._found.append(True)
        self._image_landmarks.append(image_landmarks)
        self._world_landmarks.append(world_landmarks)
        self._score.append(score)

    def add_missing(self) -> None:
        self._found.append(False)
        self._image_landmarks.append(_missing_positions(hand_landmarks.LANDMARK_COUNT))
        self._world_landmarks.append(_missing_positions(hand_landmarks.LANDMARK_COUNT))
        self._score.append(float("nan"))

    def finish(self) -> HandTrack:
        return HandTrack(
            found=np.array(self._found, dtype=bool),
            image_landmarks=np.stack(self._image_landmarks),
            world_landmarks=np.stack(self._world_landmarks),
            score=np.array(self._score, dtype=np.float32),
        )


def nearest_hand(crop_wrists: list[np.ndarray], expected_wrist: np.ndarray) -> int | None:
    if not crop_wrists:
        return None
    distances = [float(np.linalg.norm(wrist - expected_wrist)) for wrist in crop_wrists]
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] <= _FARTHEST_WRIST_SHARE_OF_CROP else None


def hand_crop_box(pose_pixels: np.ndarray, visibility: np.ndarray, side: str) -> CropBox | None:
    wrist, index, pinky = _HAND_POINTS[side]
    if not visibility[wrist] >= _SMALLEST_WRIST_VISIBILITY:
        return None
    shoulders = (pose_pixels[pose_landmarks.LEFT_SHOULDER] + pose_pixels[pose_landmarks.RIGHT_SHOULDER]) / 2.0
    hips = (pose_pixels[pose_landmarks.LEFT_HIP] + pose_pixels[pose_landmarks.RIGHT_HIP]) / 2.0
    shoulder_width = float(np.linalg.norm(pose_pixels[pose_landmarks.LEFT_SHOULDER] - pose_pixels[pose_landmarks.RIGHT_SHOULDER]))
    torso_pixels = max(float(np.linalg.norm(shoulders - hips)), 1.2 * shoulder_width)
    knuckles = (pose_pixels[index] + pose_pixels[pinky]) / 2.0
    return CropBox.around_hand(pose_pixels[wrist], knuckles, torso_pixels)


def capture_landmarks(
    video_path: Path,
    settings: CaptureSettings,
    models_folder: Path,
    progress: CaptureProgress,
    cancel_event: threading.Event,
) -> LandmarkCache:
    video_status = video_path.stat()
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise CaptureError(f"Cannot open the video {video_path}.")
    frame_rate = video.get(cv2.CAP_PROP_FPS) or _FALLBACK_FRAME_RATE
    progress.set_expected_frame_count(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    pose_recorder = _PoseRecorder()
    left_hand_recorder = _HandRecorder()
    right_hand_recorder = _HandRecorder()
    timestamps_milliseconds: list[int] = []
    first_position_milliseconds: float | None = None
    width = 0
    height = 0

    with contextlib.ExitStack() as resources:
        resources.callback(video.release)
        pose_landmarker = resources.enter_context(_create_pose_landmarker(models_folder, settings))
        hand_landmarkers = {
            "left": resources.enter_context(_create_hand_landmarker(models_folder)),
            "right": resources.enter_context(_create_hand_landmarker(models_folder)),
        }
        hand_recorders = {"left": left_hand_recorder, "right": right_hand_recorder}
        while True:
            if cancel_event.is_set():
                raise CaptureCancelled()
            frame_was_read, frame = video.read()
            if not frame_was_read:
                break
            height, width = frame.shape[:2]
            position_milliseconds = video.get(cv2.CAP_PROP_POS_MSEC)
            if first_position_milliseconds is None:
                first_position_milliseconds = position_milliseconds
            timestamp_milliseconds = int(round(position_milliseconds - first_position_milliseconds))
            if timestamps_milliseconds:
                timestamp_milliseconds = max(timestamp_milliseconds, timestamps_milliseconds[-1] + 1)
            timestamps_milliseconds.append(timestamp_milliseconds)

            rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=rgb_frame)
            pose_result = pose_landmarker.detect_for_video(image, timestamp_milliseconds)
            pose_found = bool(pose_result.pose_landmarks)
            if pose_found:
                pose_recorder.add(pose_result.pose_landmarks[0], pose_result.pose_world_landmarks[0])
            else:
                pose_recorder.add_missing()

            pose_pixels = _positions(pose_result.pose_landmarks[0])[:, :2] * (width, height) if pose_found else None
            visibility = _optional_values(landmark.visibility for landmark in pose_result.pose_landmarks[0]) if pose_found else None
            for side, recorder in hand_recorders.items():
                box = hand_crop_box(pose_pixels, visibility, side) if pose_pixels is not None and visibility is not None else None
                if box is None:
                    recorder.add_missing()
                    continue
                _track_hand_in_crop(
                    hand_landmarkers[side], recorder, box, rgb_frame, pose_pixels[_HAND_POINTS[side][0]], timestamp_milliseconds, width, height
                )
            progress.advance()

    if not timestamps_milliseconds:
        raise CaptureError(f"No frames could be read from {video_path}.")
    return LandmarkCache(
        tracking_model=tracking_models.MEDIAPIPE_POSE_HANDS,
        settings=settings,
        video=VideoSource(
            file_size=video_status.st_size,
            modified_time_nanoseconds=video_status.st_mtime_ns,
            frame_rate=frame_rate,
            width=width,
            height=height,
        ),
        frame_indices=np.arange(len(timestamps_milliseconds), dtype=np.int32),
        timestamps_milliseconds=np.array(timestamps_milliseconds, dtype=np.int64),
        pose=pose_recorder.finish(),
        left_hand=left_hand_recorder.finish(),
        right_hand=right_hand_recorder.finish(),
    )


def _create_pose_landmarker(models_folder: Path, settings: CaptureSettings) -> vision.PoseLandmarker:
    model_path = models_folder / f"pose_landmarker_{settings.pose_model_variant}.task"
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
        )
    )


def _create_hand_landmarker(models_folder: Path) -> vision.HandLandmarker:
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_folder / _HAND_MODEL_FILE_NAME)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )


def _track_hand_in_crop(
    landmarker: vision.HandLandmarker,
    recorder: _HandRecorder,
    box: CropBox,
    rgb_frame: np.ndarray,
    pose_wrist_pixels: np.ndarray,
    timestamp_milliseconds: int,
    width: int,
    height: int,
) -> None:
    crop = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=np.ascontiguousarray(box.cut(rgb_frame)))
    result = landmarker.detect_for_video(crop, timestamp_milliseconds)
    crop_wrists = [_positions(landmarks)[hand_landmarks.WRIST, :2] for landmarks in result.hand_landmarks]
    hand_index = nearest_hand(crop_wrists, box.to_crop(pose_wrist_pixels))
    if hand_index is None:
        recorder.add_missing()
        return
    recorder.add(
        box.to_frame(_positions(result.hand_landmarks[hand_index]), width, height),
        _positions(result.hand_world_landmarks[hand_index]),
        float(result.handedness[hand_index][0].score),
    )


def _positions(landmarks: list[Any]) -> np.ndarray:
    return np.array([(landmark.x, landmark.y, landmark.z) for landmark in landmarks], dtype=np.float32)


def _optional_values(values: Any) -> np.ndarray:
    return np.array([np.nan if value is None else value for value in values], dtype=np.float32)


def _missing_positions(landmark_count: int) -> np.ndarray:
    return np.full((landmark_count, 3), np.nan, dtype=np.float32)
