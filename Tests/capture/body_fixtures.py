from pathlib import Path

import numpy as np

from bl_ext.user_default.freeclipmocap.capture.capture_settings import CaptureSettings, PoseModelVariant
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import HandTrack, LandmarkCache, PoseTrack, VideoSource
from bl_ext.user_default.freeclipmocap.constants import hand_landmarks, pose_landmarks, tracking_models

from ..solver.synthetic_actor import make_rig, rest_landmarks
from .calibration_fixtures import FRAME_RATE, HEIGHT, WIDTH, SyntheticCamera, looking_camera

FRONT_CAMERA = looking_camera(np.array([0.0, -3.0, 1.2]), 1100.0)
SIDE_CAMERA = looking_camera(np.array([2.3, -1.9, 1.2]), 1250.0)
SIDE_STARTED_LATER_SECONDS = 0.37
TAKE_SECONDS = 20.0
NOISE_PIXELS = 2.0
JOINT_GUESS_METRES = 0.015
SWAPPED_SHARE = 0.03
CAPTURE_SETTINGS = CaptureSettings(pose_model_variant=PoseModelVariant.HEAVY)
_SPOT_SECONDS = 1.2
_STILL_SHARE = 2.0 / 3.0
_SPOT_RADIUS_METRES = 0.55
_ARMS = {
    side: (
        getattr(pose_landmarks, f"{side}_SHOULDER"),
        [getattr(pose_landmarks, f"{side}_{name}") for name in ("ELBOW", "WRIST", "PINKY", "INDEX", "THUMB")],
    )
    for side in ("LEFT", "RIGHT")
}
_LEGS = {
    side: (
        getattr(pose_landmarks, f"{side}_HIP"),
        [getattr(pose_landmarks, f"{side}_{name}") for name in ("KNEE", "ANKLE", "HEEL", "FOOT_INDEX")],
    )
    for side in ("LEFT", "RIGHT")
}
_MIRRORED = [
    (getattr(pose_landmarks, f"LEFT_{name}"), getattr(pose_landmarks, f"RIGHT_{name}"))
    for name in ("SHOULDER", "ELBOW", "WRIST", "PINKY", "INDEX", "THUMB", "HIP", "KNEE", "ANKLE", "HEEL", "FOOT_INDEX")
]
_rest_pose: np.ndarray | None = None


def rest_pose() -> np.ndarray:
    global _rest_pose
    if _rest_pose is None:
        rig, _bone_map = make_rig()
        _rest_pose = rest_landmarks(rig)["pose"]
    return _rest_pose


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = axis / np.linalg.norm(axis)
    cross = np.array([[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def _spot(index: int) -> tuple[np.ndarray, float]:
    angle = 2.4 * index
    return _SPOT_RADIUS_METRES * np.array([np.cos(angle), np.sin(angle), 0.0]), 0.8 * np.sin(1.3 * index)


def actor_points(time_seconds: float) -> np.ndarray:
    points = rest_pose().copy()
    segment = int(time_seconds // _SPOT_SECONDS)
    share = (time_seconds - segment * _SPOT_SECONDS) / _SPOT_SECONDS
    moving = share >= _STILL_SHARE
    blend = 0.0 if not moving else (share - _STILL_SHARE) / (1.0 - _STILL_SHARE)
    blend = blend * blend * (3.0 - 2.0 * blend)
    for side, sign in (("LEFT", 1.0), ("RIGHT", -1.0)):
        shoulder, arm = _ARMS[side]
        lowered = _rotation(np.array([0.0, 1.0, 0.0]), sign * (0.9 + 0.5 * np.sin(4.4 * time_seconds + sign)))
        swung = _rotation(np.array([1.0, 0.0, 0.0]), 0.6 * np.sin(3.1 * time_seconds + 2.0 * sign))
        points[arm] = (points[arm] - points[shoulder]) @ (swung @ lowered).T + points[shoulder]
        if moving:
            hip, leg = _LEGS[side]
            stride = _rotation(np.array([1.0, 0.0, 0.0]), sign * 0.35 * np.sin(np.pi * blend))
            points[leg] = (points[leg] - points[hip]) @ stride.T + points[hip]
    start, start_yaw = _spot(segment)
    end, end_yaw = _spot(segment + 1)
    yaw = _rotation(np.array([0.0, 0.0, 1.0]), start_yaw + blend * (end_yaw - start_yaw))
    return points @ yaw.T + start + blend * (end - start)


def filmed_cache(
    camera: SyntheticCamera,
    started_at_seconds: float,
    frame_count: int,
    seed: int,
    video_path: Path | None = None,
    swapped_share: float = 0.0,
) -> LandmarkCache:
    generator = np.random.default_rng(seed)
    joint_guesses = generator.normal(0.0, JOINT_GUESS_METRES, size=(pose_landmarks.LANDMARK_COUNT, 3))
    image = np.full((frame_count, pose_landmarks.LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    visibility = np.zeros((frame_count, pose_landmarks.LANDMARK_COUNT), dtype=np.float32)
    for frame_index in range(frame_count):
        world = actor_points(started_at_seconds + frame_index / FRAME_RATE) + joint_guesses
        camera_points = world @ camera.rotation.T + camera.translation
        pixels = camera_points[:, :2] / camera_points[:, 2:3] * camera.focal_length_pixels + (WIDTH / 2.0, HEIGHT / 2.0)
        pixels += generator.normal(0.0, NOISE_PIXELS, size=pixels.shape)
        if generator.uniform() < swapped_share:
            for left, right in _MIRRORED:
                pixels[[left, right]] = pixels[[right, left]]
        inside = (camera_points[:, 2] > 0.1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < WIDTH) & (pixels[:, 1] >= 0) & (pixels[:, 1] < HEIGHT)
        image[frame_index, :, 0] = pixels[:, 0] / WIDTH
        image[frame_index, :, 1] = pixels[:, 1] / HEIGHT
        image[frame_index, :, 2] = 0.0
        visibility[frame_index] = np.where(inside, 0.95, 0.1)
    if video_path is not None:
        status = video_path.stat()
        file_size, modified_time = status.st_size, status.st_mtime_ns
    else:
        file_size, modified_time = 1, 1
    no_hand = HandTrack(
        found=np.zeros(frame_count, dtype=bool),
        image_landmarks=np.full((frame_count, hand_landmarks.LANDMARK_COUNT, 3), np.nan, dtype=np.float32),
        world_landmarks=np.full((frame_count, hand_landmarks.LANDMARK_COUNT, 3), np.nan, dtype=np.float32),
        score=np.zeros(frame_count, dtype=np.float32),
    )
    return LandmarkCache(
        tracking_model=tracking_models.MEDIAPIPE_POSE_HANDS,
        settings=CAPTURE_SETTINGS,
        video=VideoSource(file_size, modified_time, FRAME_RATE, WIDTH, HEIGHT),
        frame_indices=np.arange(frame_count, dtype=np.int32),
        timestamps_milliseconds=np.round(np.arange(frame_count) * 1000.0 / FRAME_RATE).astype(np.int64),
        pose=PoseTrack(
            found=np.ones(frame_count, dtype=bool),
            image_landmarks=image,
            world_landmarks=np.zeros_like(image),
            visibility=visibility,
            presence=np.ones_like(visibility),
        ),
        left_hand=no_hand,
        right_hand=no_hand,
    )


def filmed_take(front_video: Path | None = None, side_video: Path | None = None) -> tuple[LandmarkCache, LandmarkCache]:
    front = filmed_cache(FRONT_CAMERA, 0.0, int(TAKE_SECONDS * FRAME_RATE), seed=11, video_path=front_video)
    side = filmed_cache(
        SIDE_CAMERA,
        SIDE_STARTED_LATER_SECONDS,
        int((TAKE_SECONDS - SIDE_STARTED_LATER_SECONDS) * FRAME_RATE),
        seed=12,
        video_path=side_video,
        swapped_share=SWAPPED_SHARE,
    )
    return front, side
