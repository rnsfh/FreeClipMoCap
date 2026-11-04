from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..capture.camera_setup import CalibratedCamera
from ..capture.landmark_cache import HandTrack, LandmarkCache, PoseTrack
from ..constants import hand_landmarks, pose_landmarks

_LARGEST_ERROR_SHARE_OF_HEIGHT = 0.03
_SWAP_IMPROVEMENT_SHARE = 0.5
_FEWEST_POINTS_TO_COMPARE = 4
_UNDISTORTION_STEPS = 20
_NEAREST_DEPTH_METRES = 0.05
_TORSO_POINTS = [
    pose_landmarks.LEFT_SHOULDER,
    pose_landmarks.RIGHT_SHOULDER,
    pose_landmarks.LEFT_HIP,
    pose_landmarks.RIGHT_HIP,
]
# The calibration's world has x to the right, y along camera 1's level view and z up. The solver reads MediaPipe's
# axes (x right, y down, z away from the camera), so the triangulated body is handed over as a level camera 1 sees it.
_WORLD_TO_TRACKER_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


class TriangulationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class HandSources:
    triangulated_share: float
    one_camera_share: float


@dataclass(frozen=True, slots=True)
class TriangulationReport:
    body_share: float
    median_error_pixels: float
    swapped_frame_counts: tuple[int, ...]
    left_hand: HandSources
    right_hand: HandSources


def triangulated_cache(
    caches: Sequence[LandmarkCache], cameras: Sequence[CalibratedCamera], start_delays_seconds: Sequence[float]
) -> tuple[LandmarkCache, TriangulationReport]:
    if len(caches) < 2 or len(caches) != len(cameras) or len(caches) != len(start_delays_seconds):
        raise TriangulationError("Triangulation needs two or more cameras, each with its capture and its calibration.")
    reference = caches[0]
    times = reference.timestamps_milliseconds.astype(np.float64) / 1000.0
    own_times = [
        times - (delay + camera.picture_delay_correction_seconds - cameras[0].picture_delay_correction_seconds)
        for delay, camera in zip(start_delays_seconds, cameras)
    ]
    pose_pixels = [_sampled(_pose_pixels(cache), cache, at) for cache, at in zip(caches, own_times)]
    left_pixels = [_sampled(_hand_pixels(cache, cache.left_hand), cache, at) for cache, at in zip(caches, own_times)]
    right_pixels = [_sampled(_hand_pixels(cache, cache.right_hand), cache, at) for cache, at in zip(caches, own_times)]
    left_worlds = [_sampled(_hand_worlds(cache.left_hand), cache, at) for cache, at in zip(caches, own_times)]
    right_worlds = [_sampled(_hand_worlds(cache.right_hand), cache, at) for cache, at in zip(caches, own_times)]

    pose_rays = [_rays(pixels, camera) for pixels, camera in zip(pose_pixels, cameras)]
    swapped_frame_counts: list[int] = []
    for camera_index in range(1, len(cameras)):
        swapped = _swapped_frames(cameras, pose_rays, camera_index)
        swapped_frame_counts.append(int(swapped.sum()))
        pose_rays[camera_index][swapped] = pose_rays[camera_index][swapped][:, _POSE_MIRROR]
        for first, second in ((left_pixels, right_pixels), (left_worlds, right_worlds)):
            first[camera_index][swapped], second[camera_index][swapped] = (
                second[camera_index][swapped].copy(),
                first[camera_index][swapped].copy(),
            )

    body, errors = _triangulated(cameras, pose_rays)
    torso_found = np.isfinite(body[:, _TORSO_POINTS, 0]).all(axis=1)
    pose = PoseTrack(
        found=torso_found,
        image_landmarks=reference.pose.image_landmarks,
        world_landmarks=(body @ _WORLD_TO_TRACKER_AXES.T).astype(np.float32),
        visibility=np.isfinite(body[..., 0]).astype(np.float32),
        presence=np.isfinite(body[..., 0]).astype(np.float32),
    )
    left_hand, left_sources = _hand_track(cameras, left_pixels, left_worlds, reference.left_hand)
    right_hand, right_sources = _hand_track(cameras, right_pixels, right_worlds, reference.right_hand)
    finite_errors = errors[np.isfinite(errors)]
    report = TriangulationReport(
        body_share=float(torso_found.mean()) if len(torso_found) else 0.0,
        median_error_pixels=float(np.median(finite_errors)) if len(finite_errors) else float("nan"),
        swapped_frame_counts=tuple(swapped_frame_counts),
        left_hand=left_sources,
        right_hand=right_sources,
    )
    return replace(reference, pose=pose, left_hand=left_hand, right_hand=right_hand), report


def _mirror_order() -> np.ndarray:
    order = np.arange(pose_landmarks.LANDMARK_COUNT)
    for name in dir(pose_landmarks):
        if "LEFT" in name:
            left = getattr(pose_landmarks, name)
            right = getattr(pose_landmarks, name.replace("LEFT", "RIGHT"))
            order[left], order[right] = right, left
    return order


_POSE_MIRROR = _mirror_order()


def _pose_pixels(cache: LandmarkCache) -> np.ndarray:
    # Every point MediaPipe placed is used, also those it marks as hidden: its guesses for a hidden elbow agreed
    # between the user's two cameras within ~5 px, and points the cameras disagree on are dropped after triangulating.
    pixels = cache.pose.image_landmarks[..., :2].astype(np.float64) * (cache.video.width, cache.video.height)
    pixels[~cache.pose.found] = np.nan
    return pixels


def _hand_pixels(cache: LandmarkCache, track: HandTrack) -> np.ndarray:
    pixels = track.image_landmarks[..., :2].astype(np.float64) * (cache.video.width, cache.video.height)
    pixels[~track.found] = np.nan
    return pixels


def _hand_worlds(track: HandTrack) -> np.ndarray:
    worlds = track.world_landmarks.astype(np.float64)
    worlds[~track.found] = np.nan
    return worlds


def _sampled(values: np.ndarray, cache: LandmarkCache, times_seconds: np.ndarray) -> np.ndarray:
    own_times = cache.timestamps_milliseconds.astype(np.float64) / 1000.0
    after = np.clip(np.searchsorted(own_times, times_seconds - 1e-6, side="left"), 0, len(own_times) - 1)
    exact = np.abs(own_times[after] - times_seconds) < 1e-6
    before = np.clip(np.where(exact, after, after - 1), 0, len(own_times) - 1)
    inside = (times_seconds >= own_times[0] - 1e-6) & (times_seconds <= own_times[-1] + 1e-6)
    consecutive = exact | (cache.frame_indices[after] - cache.frame_indices[before] == 1)
    span = np.where(after > before, own_times[after] - own_times[before], 1.0)
    share = np.where(exact, 0.0, (times_seconds - own_times[before]) / span)
    sampled = values[before] + share[:, None, None] * (values[after] - values[before])
    sampled[~(inside & consecutive)] = np.nan
    return sampled


def _rays(pixels: np.ndarray, camera: CalibratedCamera) -> np.ndarray:
    matrix = camera.camera_matrix
    distorted = (pixels - matrix[:2, 2]) / (matrix[0, 0], matrix[1, 1])
    radial_one, radial_two, tangential_one, tangential_two, radial_three = camera.distortion
    if not np.any(camera.distortion):
        return distorted
    ideal = distorted.copy()
    for _step in range(_UNDISTORTION_STEPS):
        x, y = ideal[..., 0], ideal[..., 1]
        squared_radius = x * x + y * y
        radial = 1.0 + squared_radius * (radial_one + squared_radius * (radial_two + squared_radius * radial_three))
        shift_x = 2.0 * tangential_one * x * y + tangential_two * (squared_radius + 2.0 * x * x)
        shift_y = tangential_one * (squared_radius + 2.0 * y * y) + 2.0 * tangential_two * x * y
        ideal = np.stack(((distorted[..., 0] - shift_x) / radial, (distorted[..., 1] - shift_y) / radial), axis=-1)
    return ideal


def _triangulated(cameras: Sequence[CalibratedCamera], rays_per_camera: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    shape = rays_per_camera[0].shape[:-1]
    rows = []
    seen_count = np.zeros(int(np.prod(shape)), dtype=np.int64)
    for camera, rays in zip(cameras, rays_per_camera):
        flat = rays.reshape(-1, 2)
        seen = np.isfinite(flat[:, 0])
        seen_count += seen
        flat = np.nan_to_num(flat)
        projection = np.hstack((camera.rotation, camera.translation[:, None]))
        weight = seen[:, None].astype(np.float64)
        rows.append(weight * (flat[:, 0:1] * projection[2] - projection[0]))
        rows.append(weight * (flat[:, 1:2] * projection[2] - projection[1]))
    # Linear triangulation: the point is the null vector of the stacked projection rows of every camera that saw it.
    _left, _singular_values, right = np.linalg.svd(np.stack(rows, axis=1))
    homogeneous = right[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        world = homogeneous[:, :3] / homogeneous[:, 3:4]
    world[seen_count < 2] = np.nan
    errors = np.zeros(len(world))
    for camera, rays in zip(cameras, rays_per_camera):
        flat = rays.reshape(-1, 2)
        camera_points = world @ camera.rotation.T + camera.translation
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = camera_points[:, :2] / camera_points[:, 2:3]
            error = np.linalg.norm(projected - flat, axis=-1) * camera.camera_matrix[0, 0]
        behind = camera_points[:, 2] < _NEAREST_DEPTH_METRES
        error = np.where(np.isfinite(flat[:, 0]), np.where(behind, np.inf, error), 0.0)
        too_far = error > _LARGEST_ERROR_SHARE_OF_HEIGHT * camera.image_height
        errors = np.maximum(errors, np.where(too_far, np.inf, error))
    rejected = ~np.isfinite(errors)
    world[rejected] = np.nan
    errors[rejected | (seen_count < 2)] = np.nan
    return world.reshape(*shape, 3), errors.reshape(shape)


def _swapped_frames(cameras: Sequence[CalibratedCamera], pose_rays: Sequence[np.ndarray], camera_index: int) -> np.ndarray:
    pair = [cameras[0], cameras[camera_index]]
    as_labelled = _pair_errors(pair, pose_rays[0], pose_rays[camera_index])
    mirrored = _pair_errors(pair, pose_rays[0], pose_rays[camera_index][:, _POSE_MIRROR])
    swapped = np.zeros(len(as_labelled), dtype=bool)
    for frame, (labelled_errors, mirrored_errors) in enumerate(zip(as_labelled, mirrored)):
        labelled_errors = labelled_errors[np.isfinite(labelled_errors)]
        mirrored_errors = mirrored_errors[np.isfinite(mirrored_errors)]
        if len(mirrored_errors) < _FEWEST_POINTS_TO_COMPARE:
            continue
        if len(labelled_errors) < _FEWEST_POINTS_TO_COMPARE:
            swapped[frame] = len(mirrored_errors) >= 2 * _FEWEST_POINTS_TO_COMPARE
            continue
        swapped[frame] = np.median(mirrored_errors) < _SWAP_IMPROVEMENT_SHARE * np.median(labelled_errors)
    return swapped


def _pair_errors(cameras: Sequence[CalibratedCamera], first: np.ndarray, second: np.ndarray) -> np.ndarray:
    _world, errors = _triangulated(cameras, [first, second])
    return errors


def _hand_track(
    cameras: Sequence[CalibratedCamera],
    pixels_per_camera: Sequence[np.ndarray],
    worlds_per_camera: Sequence[np.ndarray],
    reference_track: HandTrack,
) -> tuple[HandTrack, HandSources]:
    points, _errors = _triangulated(cameras, [_rays(pixels, camera) for pixels, camera in zip(pixels_per_camera, cameras)])
    triangulated = np.isfinite(points[..., 0]).all(axis=1)
    hands = np.where(triangulated[:, None, None], points - points[:, hand_landmarks.WRIST : hand_landmarks.WRIST + 1], np.nan)
    one_camera = np.zeros(len(points), dtype=bool)
    for camera, worlds in zip(cameras, worlds_per_camera):
        usable = ~triangulated & ~one_camera & np.isfinite(worlds[..., 0]).all(axis=1)
        # MediaPipe's hand points lie in the camera's own axes (x right, y down, z forward); turned into the world.
        in_world = worlds[usable] @ camera.rotation
        hands[usable] = in_world - in_world[:, hand_landmarks.WRIST : hand_landmarks.WRIST + 1]
        one_camera |= usable
    found = triangulated | one_camera
    hands[~found] = np.nan
    track = HandTrack(
        found=found,
        image_landmarks=reference_track.image_landmarks,
        world_landmarks=(hands @ _WORLD_TO_TRACKER_AXES.T).astype(np.float32),
        score=np.where(found, 1.0, np.nan).astype(np.float32),
    )
    frame_count = max(len(found), 1)
    return track, HandSources(float(triangulated.sum()) / frame_count, float(one_camera.sum()) / frame_count)
