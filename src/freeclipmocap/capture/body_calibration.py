import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from ..constants import pose_landmarks
from .camera_calibration import CalibrationError, world_frame
from .camera_setup import CalibratedCamera, CalibrationInputs, CameraSetup, FloorSource, TypedLens
from .landmark_cache import LandmarkCache
from .saved_lens import SavedLens

_BODY_POINTS = (
    pose_landmarks.LEFT_SHOULDER,
    pose_landmarks.RIGHT_SHOULDER,
    pose_landmarks.LEFT_ELBOW,
    pose_landmarks.RIGHT_ELBOW,
    pose_landmarks.LEFT_WRIST,
    pose_landmarks.RIGHT_WRIST,
    pose_landmarks.LEFT_HIP,
    pose_landmarks.RIGHT_HIP,
    pose_landmarks.LEFT_KNEE,
    pose_landmarks.RIGHT_KNEE,
    pose_landmarks.LEFT_ANKLE,
    pose_landmarks.RIGHT_ANKLE,
    pose_landmarks.LEFT_HEEL,
    pose_landmarks.RIGHT_HEEL,
    pose_landmarks.LEFT_FOOT_INDEX,
    pose_landmarks.RIGHT_FOOT_INDEX,
)
_MIRRORED_ORDER = np.arange(len(_BODY_POINTS)).reshape(-1, 2)[:, ::-1].ravel()
_SHOULDERS = [_BODY_POINTS.index(pose_landmarks.LEFT_SHOULDER), _BODY_POINTS.index(pose_landmarks.RIGHT_SHOULDER)]
_ANKLES = [_BODY_POINTS.index(pose_landmarks.LEFT_ANKLE), _BODY_POINTS.index(pose_landmarks.RIGHT_ANKLE)]
_FEET = [
    _BODY_POINTS.index(landmark)
    for landmark in (
        pose_landmarks.LEFT_HEEL,
        pose_landmarks.RIGHT_HEEL,
        pose_landmarks.LEFT_FOOT_INDEX,
        pose_landmarks.RIGHT_FOOT_INDEX,
    )
]
_SMALLEST_VISIBILITY = 0.5
_MOST_FRAMES = 600
_FEWEST_SHARED_POINTS = 150
_RANSAC_THRESHOLD_PIXELS = 4.0
_ROBUST_SCALE_PIXELS = 3.0
_SWAP_IMPROVEMENT_SHARE = 0.5
_LARGEST_PICTURE_DELAY_CORRECTION_SECONDS = 0.06
_COARSE_DELAY_STEP_SECONDS = 0.004
_FINE_DELAY_STEP_SECONDS = 0.0005
_SMALLEST_USEFUL_ERROR_DROP = 0.05
_SMALLEST_SEARCH_SPEED_PIXELS_PER_SECOND = 60.0
_MOST_REFINEMENT_ITERATIONS = 60
_DIFFERENCE_STEP = 1e-6
_LARGEST_DAMPING = 1e10
_CONVERGED_COST_SHARE = 1e-9
_NEAREST_DEPTH_METRES = 0.05
_PLANTED_SPEED_METRES_PER_SECOND = 0.15
_FEWEST_PLANTED_POINTS = 30
_SMALLEST_FLOOR_SPREAD_METRES = 0.3
_LOWEST_FOOT_SHARE = 0.05


@dataclass(frozen=True, slots=True, eq=False)
class _BodyLens:
    camera_matrix: np.ndarray
    distortion: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class _Camera:
    rotation: np.ndarray
    translation: np.ndarray
    focal_length_pixels: float
    principal_point: np.ndarray

    @property
    def camera_matrix(self) -> np.ndarray:
        focal = self.focal_length_pixels
        return np.array([[focal, 0.0, self.principal_point[0]], [0.0, focal, self.principal_point[1]], [0.0, 0.0, 1.0]])


def calibrate_cameras_from_bodies(
    caches: Sequence[LandmarkCache], video_file_names: Sequence[str], inputs: CalibrationInputs
) -> CameraSetup:
    if len(caches) < 2:
        raise CalibrationError("At least two cameras are needed.")
    if inputs.body is None:
        raise CalibrationError("These calibration settings are for the board, not the body.")
    lenses = [
        _body_lens(cache, file_name, saved_lens, typed_lens)
        for cache, file_name, saved_lens, typed_lens in zip(caches, video_file_names, inputs.saved_lenses, inputs.typed_lenses)
    ]
    reference = caches[0]
    frames = _reference_frames(reference, inputs.first_frame, inputs.stop_frame)
    times_seconds = reference.timestamps_milliseconds[frames] / 1000.0
    reference_points = _ideal_pixels(_points_at(reference, times_seconds), lenses[0])

    points_per_camera = [reference_points]
    corrections = [0.0]
    cameras = [_Camera(np.eye(3), np.zeros(3), float(lenses[0].camera_matrix[0, 0]), lenses[0].camera_matrix[:2, 2])]
    for camera_index in range(1, len(caches)):
        delay_seconds = inputs.start_delays_seconds[camera_index]
        correction, points, rotation, direction = _pair_with_reference(
            reference_points, caches[camera_index], times_seconds, delay_seconds, lenses[0], lenses[camera_index], video_file_names
        )
        if camera_index == 1:
            translation = direction * inputs.body.cameras_apart_metres
        else:
            translation, rotation = _pose_from_first_pair(points_per_camera, cameras, points, lenses[camera_index], rotation)
        lens = lenses[camera_index]
        cameras.append(_Camera(rotation, translation, float(lens.camera_matrix[0, 0]), lens.camera_matrix[:2, 2]))
        points_per_camera.append(points)
        corrections.append(correction)

    cameras = _refined(cameras, points_per_camera, inputs.body.cameras_apart_metres)
    world_points = _triangulated(cameras, points_per_camera)
    errors_per_camera = _reprojection_errors(cameras, points_per_camera, world_points)
    floor_up, floor_point = _floor_from_feet(world_points, times_seconds)
    rotations = [camera.rotation for camera in cameras]
    translations = [camera.translation for camera in cameras]
    world_rotation, world_origin = world_frame(rotations, translations, floor_up, floor_point)

    calibrated = tuple(
        CalibratedCamera(
            video_file_name=file_name,
            image_width=cache.video.width,
            image_height=cache.video.height,
            camera_matrix=camera.camera_matrix,
            distortion=lens.distortion,
            rotation=camera.rotation @ world_rotation,
            translation=camera.rotation @ world_origin + camera.translation,
            lens_error_pixels=errors,
            picture_delay_correction_seconds=correction,
        )
        for cache, file_name, lens, camera, errors, correction in zip(
            caches, video_file_names, lenses, cameras, errors_per_camera, corrections
        )
    )
    return CameraSetup(
        inputs=inputs,
        cameras=calibrated,
        floor_source=FloorSource.FEET,
        floor_camera_index=None,
        floor_video_frames=None,
        pair_error_pixels=tuple(errors_per_camera[1:]),
    )


def _body_lens(cache: LandmarkCache, file_name: str, saved_lens: SavedLens | None, typed_lens: TypedLens | None) -> _BodyLens:
    width, height = cache.video.width, cache.video.height
    if saved_lens is not None:
        if (saved_lens.image_width, saved_lens.image_height) != (width, height):
            raise CalibrationError(
                f"{file_name} is {width}×{height}, but the saved lens {saved_lens.name} is for "
                f"{saved_lens.image_width}×{saved_lens.image_height} video."
            )
        return _BodyLens(saved_lens.camera_matrix, saved_lens.distortion)
    if typed_lens is None:
        raise CalibrationError(f"{file_name}: calibrating from the body needs the lens. Type its focal length or pick a saved lens.")
    # The lens stays as typed: with the phones' view axes roughly in one plane (both aimed at the actor from about
    # the same height) the body's views fit equally well at very different focal lengths.
    focal = typed_lens.focal_length_pixels(width, height)
    camera_matrix = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
    return _BodyLens(camera_matrix, np.zeros(5))


def _reference_frames(reference: LandmarkCache, first_frame: int, stop_frame: int) -> np.ndarray:
    in_range = reference.frame_indices >= first_frame
    if stop_frame > 0:
        in_range &= reference.frame_indices < stop_frame
    frames = np.flatnonzero(in_range & reference.pose.found)
    if len(frames) > _MOST_FRAMES:
        frames = frames[np.linspace(0, len(frames) - 1, _MOST_FRAMES).round().astype(int)]
    return frames


def _points_at(cache: LandmarkCache, times_seconds: np.ndarray) -> np.ndarray:
    own_times = cache.timestamps_milliseconds / 1000.0
    pixels = cache.pose.image_landmarks[:, _BODY_POINTS, :2].astype(np.float64) * (cache.video.width, cache.video.height)
    seen = cache.pose.found[:, None] & (np.nan_to_num(cache.pose.visibility[:, _BODY_POINTS]) >= _SMALLEST_VISIBILITY)
    pixels[~seen] = np.nan
    after = np.clip(np.searchsorted(own_times, times_seconds - 1e-6, side="left"), 0, len(own_times) - 1)
    exact = np.abs(own_times[after] - times_seconds) < 1e-6
    before = np.clip(np.where(exact, after, after - 1), 0, len(own_times) - 1)
    inside = (times_seconds >= own_times[0] - 1e-6) & (times_seconds <= own_times[-1] + 1e-6)
    consecutive = exact | (cache.frame_indices[after] - cache.frame_indices[before] == 1)
    span = np.where(after > before, own_times[after] - own_times[before], 1.0)
    share = np.where(exact, 0.0, (times_seconds - own_times[before]) / span)
    points = pixels[before] + share[:, None, None] * (pixels[after] - pixels[before])
    points[~(inside & consecutive)] = np.nan
    return points


def _ideal_pixels(points: np.ndarray, lens: _BodyLens) -> np.ndarray:
    if not np.any(lens.distortion):
        return points
    finite = np.isfinite(points[..., 0])
    ideal = np.full_like(points, np.nan)
    undistorted = cv2.undistortPoints(
        points[finite].reshape(-1, 1, 2), lens.camera_matrix, lens.distortion, P=lens.camera_matrix
    )
    ideal[finite] = undistorted.reshape(-1, 2)
    return ideal


def _normalized(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    return (points - camera_matrix[:2, 2]) / camera_matrix[0, 0]


def _pair_with_reference(
    reference_points: np.ndarray,
    other: LandmarkCache,
    times_seconds: np.ndarray,
    delay_seconds: float,
    reference_lens: _BodyLens,
    other_lens: _BodyLens,
    video_file_names: Sequence[str],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    def other_points(correction: float) -> np.ndarray:
        return _ideal_pixels(_points_at(other, times_seconds - delay_seconds - correction), other_lens)

    focal = (reference_lens.camera_matrix[0, 0] + other_lens.camera_matrix[0, 0]) / 2.0
    points = other_points(0.0)
    rotation, direction = _relative_pose(reference_points, points, reference_lens, other_lens, video_file_names)
    essential = _essential(rotation, direction)
    # Only a moving body shows a timing error; a still one looks the same at every offset.
    moving = _speeds(reference_points, times_seconds) >= _SMALLEST_SEARCH_SPEED_PIXELS_PER_SECOND

    def disagreement(correction: float) -> float:
        shifted = other_points(correction)
        errors = _epipolar_errors(essential, reference_points, shifted, reference_lens, other_lens) * focal
        errors = errors[moving & np.isfinite(errors)]
        return float(np.median(errors)) if len(errors) >= _FEWEST_SHARED_POINTS else np.inf

    audio_disagreement = disagreement(0.0)
    best_correction, best_disagreement = 0.0, audio_disagreement
    if np.isfinite(audio_disagreement):
        for step, span in (
            (_COARSE_DELAY_STEP_SECONDS, _LARGEST_PICTURE_DELAY_CORRECTION_SECONDS),
            (_FINE_DELAY_STEP_SECONDS, _COARSE_DELAY_STEP_SECONDS),
        ):
            center = best_correction
            for correction in center + np.arange(-span, span + step / 2.0, step):
                candidate = disagreement(float(correction))
                if candidate < best_disagreement:
                    best_correction, best_disagreement = float(correction), candidate
        if best_disagreement > audio_disagreement * (1.0 - _SMALLEST_USEFUL_ERROR_DROP):
            best_correction = 0.0

    points = _with_left_and_right_matched(essential, reference_points, other_points(best_correction), reference_lens, other_lens)
    rotation, direction = _relative_pose(reference_points, points, reference_lens, other_lens, video_file_names)
    return best_correction, points, rotation, direction


def _relative_pose(
    first_points: np.ndarray,
    second_points: np.ndarray,
    first_lens: _BodyLens,
    second_lens: _BodyLens,
    video_file_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    shared = np.isfinite(first_points[..., 0]) & np.isfinite(second_points[..., 0])
    if shared.sum() < _FEWEST_SHARED_POINTS:
        raise CalibrationError(
            f"{video_file_names[0]} and another camera saw the body at the same time in too few frames "
            f"({int(shared.sum())} joints, at least {_FEWEST_SHARED_POINTS} needed). Film the actor where both phones see "
            "them, or check the sync offset."
        )
    first = _normalized(first_points[shared], first_lens.camera_matrix)
    second = _normalized(second_points[shared], second_lens.camera_matrix)
    focal = (first_lens.camera_matrix[0, 0] + second_lens.camera_matrix[0, 0]) / 2.0
    essential, inliers = cv2.findEssentialMat(
        first, second, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=_RANSAC_THRESHOLD_PIXELS / focal
    )
    if essential is None or essential.shape[0] < 3:
        raise CalibrationError("The cameras' views of the body do not fit together; check the sync offset and the lenses.")
    _inlier_count, rotation, translation, _mask = cv2.recoverPose(essential[:3], first, second, np.eye(3), mask=inliers)
    return rotation, translation.reshape(3) / np.linalg.norm(translation)


def _essential(rotation: np.ndarray, direction: np.ndarray) -> np.ndarray:
    x, y, z = direction
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]) @ rotation


def _epipolar_errors(
    essential: np.ndarray, first_points: np.ndarray, second_points: np.ndarray, first_lens: _BodyLens, second_lens: _BodyLens
) -> np.ndarray:
    first = _normalized(first_points, first_lens.camera_matrix)
    second = _normalized(second_points, second_lens.camera_matrix)
    first_homogeneous = np.concatenate((first, np.ones(first.shape[:-1] + (1,))), axis=-1)
    second_homogeneous = np.concatenate((second, np.ones(second.shape[:-1] + (1,))), axis=-1)
    first_lines = first_homogeneous @ essential.T
    second_lines = second_homogeneous @ essential
    # Sampson distance: the first-order distance of the point pair from satisfying the epipolar constraint.
    residual = np.sum(second_homogeneous * first_lines, axis=-1)
    scale = np.sqrt(first_lines[..., 0] ** 2 + first_lines[..., 1] ** 2 + second_lines[..., 0] ** 2 + second_lines[..., 1] ** 2)
    return np.abs(residual) / scale


def _with_left_and_right_matched(
    essential: np.ndarray, reference_points: np.ndarray, points: np.ndarray, reference_lens: _BodyLens, other_lens: _BodyLens
) -> np.ndarray:
    as_labelled = _epipolar_errors(essential, reference_points, points, reference_lens, other_lens)
    mirrored = _epipolar_errors(essential, reference_points, points[:, _MIRRORED_ORDER], reference_lens, other_lens)
    matched = points.copy()
    for frame in range(len(points)):
        labelled_errors = as_labelled[frame][np.isfinite(as_labelled[frame])]
        mirrored_errors = mirrored[frame][np.isfinite(mirrored[frame])]
        if len(labelled_errors) < 4 or len(mirrored_errors) < 4:
            continue
        if np.median(mirrored_errors) < _SWAP_IMPROVEMENT_SHARE * np.median(labelled_errors):
            matched[frame] = points[frame, _MIRRORED_ORDER]
    return matched


def _speeds(points: np.ndarray, times_seconds: np.ndarray) -> np.ndarray:
    speeds = np.full(points.shape[:2], np.nan)
    steps = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    durations = np.maximum(np.diff(times_seconds), 1e-6)[:, None]
    speeds[1:] = steps / durations
    return np.nan_to_num(speeds, nan=0.0)


def _pose_from_first_pair(
    points_per_camera: list[np.ndarray],
    cameras: list[_Camera],
    points: np.ndarray,
    lens: _BodyLens,
    rotation_guess: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_points = _triangulated(cameras[:2], points_per_camera[:2])
    usable = np.isfinite(world_points[..., 0]) & np.isfinite(points[..., 0])
    if usable.sum() < _FEWEST_SHARED_POINTS:
        raise CalibrationError("A further camera saw too few of the joints that the first two cameras saw.")
    solved, rotation_vector, translation, _inliers = cv2.solvePnPRansac(
        world_points[usable], points[usable], lens.camera_matrix, None, reprojectionError=_RANSAC_THRESHOLD_PIXELS
    )
    if not solved:
        return np.zeros(3), rotation_guess
    rotation, _jacobian = cv2.Rodrigues(rotation_vector)
    return translation.reshape(3), rotation


def _triangulated(cameras: Sequence[_Camera], points_per_camera: Sequence[np.ndarray]) -> np.ndarray:
    frame_count, point_count = points_per_camera[0].shape[:2]
    rows = []
    for camera, points in zip(cameras, points_per_camera):
        normalized = (points.reshape(-1, 2) - camera.principal_point) / camera.focal_length_pixels
        seen = np.isfinite(normalized[:, 0])
        normalized = np.nan_to_num(normalized)
        projection = np.hstack((camera.rotation, camera.translation[:, None]))
        weight = seen[:, None].astype(np.float64)
        rows.append(weight * (normalized[:, 0:1] * projection[2] - projection[0]))
        rows.append(weight * (normalized[:, 1:2] * projection[2] - projection[1]))
    seen_count = sum(np.isfinite(points.reshape(-1, 2)[:, 0]).astype(int) for points in points_per_camera)
    # Linear triangulation: the 3D point is the null vector of the stacked projection rows of every camera that saw it.
    _left, _singular_values, right = np.linalg.svd(np.stack(rows, axis=1))
    homogeneous = right[:, -1, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        world = homogeneous[:, :3] / homogeneous[:, 3:4]
    world[seen_count < 2] = np.nan
    return world.reshape(frame_count, point_count, 3)


def _projected(camera: _Camera, world_points: np.ndarray) -> np.ndarray:
    camera_points = world_points @ camera.rotation.T + camera.translation
    depths = np.maximum(camera_points[..., 2:3], _NEAREST_DEPTH_METRES)
    return camera_points[..., :2] / depths * camera.focal_length_pixels + camera.principal_point


def _reprojection_errors(cameras: Sequence[_Camera], points_per_camera: Sequence[np.ndarray], world_points: np.ndarray) -> list[float]:
    errors = []
    for camera, points in zip(cameras, points_per_camera):
        lengths = np.linalg.norm(_projected(camera, world_points) - points, axis=-1)
        lengths = lengths[np.isfinite(lengths)]
        errors.append(float(np.median(lengths)) if len(lengths) else np.inf)
    return errors


def _refined(cameras: list[_Camera], points_per_camera: list[np.ndarray], cameras_apart_metres: float) -> list[_Camera]:
    second_direction = cameras[1].translation / np.linalg.norm(cameras[1].translation)
    tangent_one = np.cross(second_direction, [1.0, 0.0, 0.0] if abs(second_direction[0]) < 0.9 else [0.0, 1.0, 0.0])
    tangent_one /= np.linalg.norm(tangent_one)
    tangent_two = np.cross(second_direction, tangent_one)

    # Parameters: each further camera's rotation change (3), the second camera's direction change on the sphere of
    # the measured distance (2), and every later camera's translation (3).
    def cameras_from(parameters: np.ndarray) -> list[_Camera]:
        rebuilt = [cameras[0]]
        offset = 0
        for index in range(1, len(cameras)):
            rotation_change, _jacobian = cv2.Rodrigues(parameters[offset : offset + 3])
            offset += 3
            if index == 1:
                direction = second_direction + parameters[offset] * tangent_one + parameters[offset + 1] * tangent_two
                translation = cameras_apart_metres * direction / np.linalg.norm(direction)
                offset += 2
            else:
                translation = cameras[index].translation + parameters[offset : offset + 3]
                offset += 3
            camera = cameras[index]
            rebuilt.append(_Camera(rotation_change @ camera.rotation, translation, camera.focal_length_pixels, camera.principal_point))
        return rebuilt

    def residuals(parameters: np.ndarray) -> np.ndarray:
        candidate = cameras_from(parameters)
        world_points = _triangulated(candidate, points_per_camera)
        lengths = []
        for camera, points in zip(candidate, points_per_camera):
            errors = np.linalg.norm(_projected(camera, world_points) - points, axis=-1)
            lengths.append(np.where(np.isfinite(errors), errors, 0.0).ravel())
        lengths = np.concatenate(lengths)
        # Cauchy loss as a residual: s * sqrt(log(1 + (e / s)²)), so mismatched joints barely pull.
        return _ROBUST_SCALE_PIXELS * np.sqrt(np.log1p(np.square(lengths / _ROBUST_SCALE_PIXELS)))

    parameter_count = 3 * (len(cameras) - 1) + 2 + 3 * (len(cameras) - 2)
    parameters = _least_squares(residuals, np.zeros(parameter_count))
    return cameras_from(parameters)


def _least_squares(residuals: Callable[[np.ndarray], np.ndarray], parameters: np.ndarray) -> np.ndarray:
    current = residuals(parameters)
    cost = float(current @ current)
    damping = 1e-3
    for _iteration in range(_MOST_REFINEMENT_ITERATIONS):
        jacobian = np.empty((len(current), len(parameters)))
        for column in range(len(parameters)):
            shifted = parameters.copy()
            shifted[column] += _DIFFERENCE_STEP
            jacobian[:, column] = (residuals(shifted) - current) / _DIFFERENCE_STEP
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        improved = False
        while damping < _LARGEST_DAMPING:
            damped = normal_matrix + damping * np.diag(np.diag(normal_matrix) + 1e-12)
            step = np.linalg.solve(damped, -gradient)
            candidate = residuals(parameters + step)
            candidate_cost = float(candidate @ candidate)
            if candidate_cost < cost:
                converged = cost - candidate_cost < _CONVERGED_COST_SHARE * cost
                parameters, current, cost = parameters + step, candidate, candidate_cost
                damping = max(damping / 3.0, 1e-12)
                improved = True
                break
            damping *= 4.0
        if not improved or converged:
            break
    return parameters


def _floor_from_feet(world_points: np.ndarray, times_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shoulders = np.nanmean(world_points[:, _SHOULDERS], axis=1)
        ankles = np.nanmean(world_points[:, _ANKLES], axis=1)
    body_up = shoulders - ankles
    body_up = body_up[np.isfinite(body_up).all(axis=1)]
    if len(body_up) == 0:
        raise CalibrationError("The body's shoulders and ankles were never seen by two cameras at once.")
    up = np.mean(body_up / np.linalg.norm(body_up, axis=1, keepdims=True), axis=0)
    up /= np.linalg.norm(up)

    feet = world_points[:, _FEET]
    speeds = np.full(feet.shape[:2], np.inf)
    durations = np.maximum(np.diff(times_seconds), 1e-6)[:, None]
    steps = np.linalg.norm(feet[1:] - feet[:-1], axis=-1) / durations
    speeds[1:] = np.where(np.isfinite(steps), steps, np.inf)
    planted = feet[speeds < _PLANTED_SPEED_METRES_PER_SECOND]
    if len(planted) >= _FEWEST_PLANTED_POINTS:
        center = planted.mean(axis=0)
        _values, spreads, axes = np.linalg.svd(planted - center, full_matrices=False)
        if spreads[1] / np.sqrt(len(planted)) >= _SMALLEST_FLOOR_SPREAD_METRES / 2.0:
            normal = axes[2] if axes[2] @ up > 0.0 else -axes[2]
            return normal, center
        return up, center
    all_feet = feet[np.isfinite(feet).all(axis=-1)]
    heights = all_feet @ up
    lowest = all_feet[heights <= np.quantile(heights, _LOWEST_FOOT_SHARE)]
    return up, lowest.mean(axis=0)
