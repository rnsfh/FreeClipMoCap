import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..constants import calibration_board
from .board_detection import board_corner_positions
from .board_sightings import BoardSightings
from .camera_setup import CalibratedCamera, CameraSetup, FloorSource

_FEWEST_CORNERS_FOR_LENS = 8
_FEWEST_LENS_FRAMES = 10
_MOST_LENS_FRAMES = 40
_LENS_FLAGS = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
_OUTLIER_SHARE_OF_MEDIAN = 3.0
_SMALLEST_OUTLIER_PIXELS = 1.5
_FEWEST_SHARED_CORNERS = 6
_FEWEST_PAIRS = 8
_MOST_SEARCH_PAIRS = 150
_MOST_STEREO_PAIRS = 40
_STEREO_FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS | _LENS_FLAGS
_STEREO_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
_LARGEST_PICTURE_DELAY_CORRECTION_SECONDS = 0.06
_COARSE_DELAY_STEP_SECONDS = 0.004
_FINE_DELAY_STEP_SECONDS = 0.0005
_SMALLEST_USEFUL_ERROR_DROP = 0.05
_SMALLEST_SEARCH_SPEED_PIXELS_PER_SECOND = 60.0
_FLOOR_STILL_PIXELS = 0.4
_FLOOR_STILL_SECONDS = 1.0
_FLOOR_FEWEST_CORNERS = 8
_FLOOR_LARGEST_TILT_DEGREES = 50.0
_FLOOR_SMALLEST_CAMERA_HEIGHT_METRES = 0.3
_FALLBACK_CENTER_DISTANCE_METRES = 2.5
_CAMERA_UP = np.array([0.0, -1.0, 0.0])
_CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])


class CalibrationError(Exception):
    pass


@dataclass(frozen=True, slots=True, eq=False)
class LensCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    error_pixels: float
    frames_used: int


@dataclass(frozen=True, slots=True, eq=False)
class _BoardViews:
    object_points: list[np.ndarray]
    first_image_points: list[np.ndarray]
    second_image_points: list[np.ndarray]

    def kept(self, keep: np.ndarray) -> "_BoardViews":
        indices = np.flatnonzero(keep)
        return _BoardViews(
            [self.object_points[index] for index in indices],
            [self.first_image_points[index] for index in indices],
            [self.second_image_points[index] for index in indices],
        )


@dataclass(frozen=True, slots=True, eq=False)
class _StereoFit:
    error_pixels: float
    rotation: np.ndarray
    translation: np.ndarray
    per_pair_errors: np.ndarray
    first_camera_matrix: np.ndarray
    first_distortion: np.ndarray
    second_camera_matrix: np.ndarray
    second_distortion: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class _FloorSighting:
    source: FloorSource
    normal: np.ndarray
    point: np.ndarray
    camera_height: float
    camera_index: int
    video_frames: tuple[int, int]


def calibrate_lens(sightings: BoardSightings, square_size_metres: float) -> LensCalibration:
    object_corners = board_corner_positions(square_size_metres).astype(np.float32)
    good_frames = np.flatnonzero(sightings.corners_seen >= _FEWEST_CORNERS_FOR_LENS)
    if len(good_frames) < _FEWEST_LENS_FRAMES:
        raise CalibrationError(
            f"{sightings.video_file_name}: the board was seen clearly in only {len(good_frames)} frames "
            f"(at least {_FEWEST_LENS_FRAMES} are needed). Bring it closer to this camera and tilt it."
        )
    chosen = good_frames[np.unique(np.linspace(0, len(good_frames) - 1, min(len(good_frames), _MOST_LENS_FRAMES)).round().astype(int))]
    object_points, image_points = [], []
    for frame in chosen:
        seen = np.isfinite(sightings.corner_points[frame, :, 0])
        object_points.append(object_corners[seen])
        image_points.append(sightings.corner_points[frame, seen].astype(np.float32))
    image_size = (sightings.image_width, sightings.image_height)

    error, camera_matrix, distortion, per_view_errors = _calibrate_lens_views(object_points, image_points, image_size)
    keep = per_view_errors <= max(_OUTLIER_SHARE_OF_MEDIAN * float(np.median(per_view_errors)), _SMALLEST_OUTLIER_PIXELS)
    if not keep.all() and keep.sum() >= _FEWEST_LENS_FRAMES:
        object_points = [points for points, kept in zip(object_points, keep) if kept]
        image_points = [points for points, kept in zip(image_points, keep) if kept]
        error, camera_matrix, distortion, _per_view_errors = _calibrate_lens_views(object_points, image_points, image_size)
    return LensCalibration(camera_matrix, distortion, error, len(object_points))


def calibrate_cameras(
    sightings_per_camera: Sequence[BoardSightings],
    start_delays_seconds: Sequence[float],
    square_size_metres: float,
    paper_height_metres: float,
) -> CameraSetup:
    if len(sightings_per_camera) < 2:
        raise CalibrationError("At least two cameras are needed.")
    object_corners = board_corner_positions(square_size_metres)
    lenses = [calibrate_lens(sightings, square_size_metres) for sightings in sightings_per_camera]
    reference = sightings_per_camera[0]

    relative_rotations = [np.eye(3)]
    relative_translations = [np.zeros(3)]
    corrections = [0.0]
    pair_errors: list[float] = []
    refined_lenses = list(lenses)
    for camera_index in range(1, len(sightings_per_camera)):
        other = sightings_per_camera[camera_index]
        correction, fit = _fit_pair(reference, other, start_delays_seconds[camera_index], lenses[0], lenses[camera_index], object_corners)
        relative_rotations.append(fit.rotation)
        relative_translations.append(fit.translation)
        corrections.append(correction)
        pair_errors.append(fit.error_pixels)
        refined_lenses[camera_index] = replace(lenses[camera_index], camera_matrix=fit.second_camera_matrix, distortion=fit.second_distortion)
        if camera_index == 1:
            refined_lenses[0] = replace(lenses[0], camera_matrix=fit.first_camera_matrix, distortion=fit.first_distortion)
    lenses = refined_lenses

    floor_sightings = [
        floor
        for camera_index, (sightings, lens) in enumerate(zip(sightings_per_camera, lenses))
        for floor in _floor_sightings(
            sightings,
            lens,
            object_corners,
            square_size_metres,
            paper_height_metres,
            camera_index,
            relative_rotations[camera_index],
            relative_translations[camera_index],
        )
    ]
    floor = max(floor_sightings, key=lambda sighting: sighting.camera_height) if floor_sightings else None
    world_rotation, world_origin = _world_frame(relative_rotations, relative_translations, floor)

    cameras = tuple(
        CalibratedCamera(
            video_file_name=sightings.video_file_name,
            image_width=sightings.image_width,
            image_height=sightings.image_height,
            camera_matrix=lens.camera_matrix,
            distortion=lens.distortion,
            rotation=rotation @ world_rotation,
            translation=rotation @ world_origin + translation,
            lens_error_pixels=lens.error_pixels,
            picture_delay_correction_seconds=correction,
        )
        for sightings, lens, rotation, translation, correction in zip(
            sightings_per_camera, lenses, relative_rotations, relative_translations, corrections
        )
    )
    return CameraSetup(
        cameras=cameras,
        square_size_metres=square_size_metres,
        floor_source=FloorSource.NONE if floor is None else floor.source,
        floor_camera_index=None if floor is None else floor.camera_index,
        floor_video_frames=None if floor is None else floor.video_frames,
        pair_error_pixels=tuple(pair_errors),
    )


def _calibrate_lens_views(
    object_points: list[np.ndarray], image_points: list[np.ndarray], image_size: tuple[int, int]
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    starting_matrix = np.array([[1.0, 0.0, image_size[0] / 2.0], [0.0, 1.0, image_size[1] / 2.0], [0.0, 0.0, 1.0]])
    error, camera_matrix, distortion, *_poses_and_deviations, per_view_errors = cv2.calibrateCameraExtended(
        object_points, image_points, image_size, starting_matrix, np.zeros(5), flags=_LENS_FLAGS
    )
    return float(error), camera_matrix, distortion.reshape(5), np.asarray(per_view_errors).ravel()


def _fit_pair(
    reference: BoardSightings,
    other: BoardSightings,
    audio_delay_seconds: float,
    reference_lens: LensCalibration,
    other_lens: LensCalibration,
    object_corners: np.ndarray,
) -> tuple[float, _StereoFit]:
    if _paired_views(reference, other, audio_delay_seconds, object_corners, _MOST_STEREO_PAIRS) is None:
        raise CalibrationError(
            f"{reference.video_file_name} and {other.video_file_name} saw the board at the same time in fewer than "
            f"{_FEWEST_PAIRS} frames. Hold it where both cameras see it, or check the sync offset."
        )
    # Only a moving board shows a timing error; a still one looks the same at every offset.
    moving = reference.corner_speeds_pixels_per_second() >= _SMALLEST_SEARCH_SPEED_PIXELS_PER_SECOND

    def disagreement(correction: float) -> float:
        views = _paired_views(reference, other, audio_delay_seconds + correction, object_corners, _MOST_SEARCH_PAIRS, moving)
        return _pose_disagreement(views, reference_lens, other_lens) if views is not None else np.inf

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

    views = _paired_views(reference, other, audio_delay_seconds + best_correction, object_corners, _MOST_STEREO_PAIRS)
    if views is None:
        raise CalibrationError(f"{reference.video_file_name} and {other.video_file_name} share too few board sightings.")
    fit = _stereo_fit(views, reference_lens, other_lens)
    keep = fit.per_pair_errors <= max(_OUTLIER_SHARE_OF_MEDIAN * float(np.median(fit.per_pair_errors)), _SMALLEST_OUTLIER_PIXELS)
    if not keep.all() and keep.sum() >= _FEWEST_PAIRS:
        fit = _stereo_fit(views.kept(keep), reference_lens, other_lens)
    return best_correction, fit


def _paired_views(
    reference: BoardSightings,
    other: BoardSightings,
    other_started_later_seconds: float,
    object_corners: np.ndarray,
    most_pairs: int,
    frame_mask: np.ndarray | None = None,
) -> _BoardViews | None:
    other_points = other.interpolated_at(reference.timestamps_seconds - other_started_later_seconds)
    shared = np.isfinite(reference.corner_points[:, :, 0]) & np.isfinite(other_points[:, :, 0])
    if frame_mask is not None:
        shared &= frame_mask[:, None]
    frames = np.flatnonzero(shared.sum(axis=1) >= _FEWEST_SHARED_CORNERS)
    if len(frames) < _FEWEST_PAIRS:
        return None
    if len(frames) > most_pairs:
        frames = frames[np.linspace(0, len(frames) - 1, most_pairs).round().astype(int)]
    return _BoardViews(
        [object_corners[shared[frame]].astype(np.float32) for frame in frames],
        [reference.corner_points[frame, shared[frame]].astype(np.float32) for frame in frames],
        [other_points[frame, shared[frame]].astype(np.float32) for frame in frames],
    )


def _pose_disagreement(views: _BoardViews, first_lens: LensCalibration, second_lens: LensCalibration) -> float:
    first_poses = [
        _board_pose(object_points, image_points, first_lens)
        for object_points, image_points in zip(views.object_points, views.first_image_points)
    ]
    second_poses = [
        _board_pose(object_points, image_points, second_lens)
        for object_points, image_points in zip(views.object_points, views.second_image_points)
    ]
    relative_rotations = np.array([second[0] @ first[0].T for first, second in zip(first_poses, second_poses)])
    relative_translations = np.array([
        second[1] - relative_rotation @ first[1]
        for first, second, relative_rotation in zip(first_poses, second_poses, relative_rotations)
    ])
    # Mean rotation: the rotation nearest (Frobenius) to the sum of the per-pair rotations.
    left, _singular_values, right = np.linalg.svd(relative_rotations.sum(axis=0))
    mean_rotation = left @ np.diag([1.0, 1.0, np.linalg.det(left @ right)]) @ right
    median_translation = np.median(relative_translations, axis=0)
    errors = []
    for (first_rotation, first_translation), object_points, image_points in zip(
        first_poses, views.object_points, views.second_image_points
    ):
        rotation_vector, _jacobian = cv2.Rodrigues(mean_rotation @ first_rotation)
        projected, _jacobian = cv2.projectPoints(
            object_points,
            rotation_vector,
            mean_rotation @ first_translation + median_translation,
            second_lens.camera_matrix,
            second_lens.distortion,
        )
        errors.append(np.sqrt(np.mean(np.sum(np.square(projected.reshape(-1, 2) - image_points), axis=1))))
    return float(np.median(errors))


def _board_pose(object_points: np.ndarray, image_points: np.ndarray, lens: LensCalibration) -> tuple[np.ndarray, np.ndarray]:
    _solved, rotation_vector, translation = cv2.solvePnP(
        object_points, image_points.astype(np.float64), lens.camera_matrix, lens.distortion, flags=cv2.SOLVEPNP_IPPE
    )
    rotation, _jacobian = cv2.Rodrigues(rotation_vector)
    return rotation, translation.reshape(3)


def _stereo_fit(views: _BoardViews, first_lens: LensCalibration, second_lens: LensCalibration) -> _StereoFit:
    (
        error,
        first_matrix,
        first_distortion,
        second_matrix,
        second_distortion,
        rotation,
        translation,
        _essential,
        _fundamental,
        _board_rotations,
        _board_translations,
        per_view_errors,
    ) = cv2.stereoCalibrateExtended(
        views.object_points,
        views.first_image_points,
        views.second_image_points,
        first_lens.camera_matrix.copy(),
        first_lens.distortion.copy(),
        second_lens.camera_matrix.copy(),
        second_lens.distortion.copy(),
        (1, 1),
        np.eye(3),
        np.zeros(3),
        flags=_STEREO_FLAGS,
        criteria=_STEREO_CRITERIA,
    )
    return _StereoFit(
        error_pixels=float(error),
        rotation=rotation,
        translation=translation.reshape(3),
        per_pair_errors=np.asarray(per_view_errors).max(axis=1),
        first_camera_matrix=first_matrix,
        first_distortion=first_distortion.reshape(5),
        second_camera_matrix=second_matrix,
        second_distortion=second_distortion.reshape(5),
    )


def _floor_sightings(
    sightings: BoardSightings,
    lens: LensCalibration,
    object_corners: np.ndarray,
    square_size_metres: float,
    paper_height_metres: float,
    camera_index: int,
    rotation_from_reference: np.ndarray,
    translation_from_reference: np.ndarray,
) -> list[_FloorSighting]:
    points = sightings.corner_points
    movement = np.full(len(points), np.inf)
    steps = np.linalg.norm(points[1:] - points[:-1], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        step_medians = np.nanmedian(np.where(np.isfinite(steps), steps, np.nan), axis=1)
    consecutive = np.diff(sightings.frame_indices) == 1
    movement[1:] = np.where(consecutive & np.isfinite(step_medians), step_medians, np.inf)
    still = (movement < _FLOOR_STILL_PIXELS) & (sightings.corners_seen >= _FLOOR_FEWEST_CORNERS)
    fewest_frames = int(np.ceil(_FLOOR_STILL_SECONDS * sightings.frame_rate))

    found = []
    for first, stop in _runs(still):
        first = max(first - 1, 0)
        if stop - first < fewest_frames:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_corners = np.nanmean(points[first:stop], axis=0)
        seen = np.isfinite(mean_corners[:, 0])
        board_rotation, translation = _board_pose(object_corners[seen], mean_corners[seen], lens)
        floor = _floor_from_still_board(board_rotation, translation, square_size_metres, paper_height_metres)
        if floor is None:
            continue
        source, up, point = floor
        reference_up = rotation_from_reference.T @ up
        reference_point = rotation_from_reference.T @ (point - translation_from_reference)
        found.append(
            _FloorSighting(
                source=source,
                normal=reference_up,
                point=reference_point,
                camera_height=float(reference_up @ -reference_point),
                camera_index=camera_index,
                video_frames=(int(sightings.frame_indices[first]), int(sightings.frame_indices[stop - 1])),
            )
        )
    return found


def _floor_from_still_board(
    board_rotation: np.ndarray, translation: np.ndarray, square_size_metres: float, paper_height_metres: float
) -> tuple[FloorSource, np.ndarray, np.ndarray] | None:
    board_width = calibration_board.SQUARES_ACROSS * square_size_metres
    board_height = calibration_board.SQUARES_DOWN * square_size_metres
    largest_tilt_cosine = np.cos(np.radians(_FLOOR_LARGEST_TILT_DEGREES))
    normal = board_rotation[:, 2]
    center = board_rotation @ np.array([board_width / 2.0, board_height / 2.0, 0.0]) + translation
    if normal @ -center < 0.0:
        normal = -normal
    board_down = board_rotation[:, 1]
    if normal @ _CAMERA_UP >= largest_tilt_cosine:
        source, up, point = FloorSource.BOARD_FLAT, normal, center
    elif abs(board_down @ -_CAMERA_UP) >= largest_tilt_cosine:
        # The board is printed centred on the page, so the paper's bottom edge lies one margin below the squares,
        # whichever way up the board stands.
        margin = (paper_height_metres - board_height) / 2.0
        standing_upright = board_down @ -_CAMERA_UP > 0.0
        edge_height = board_height + margin if standing_upright else -margin
        up = -board_down if standing_upright else board_down
        source, point = FloorSource.BOARD_UPRIGHT, board_rotation @ np.array([board_width / 2.0, edge_height, 0.0]) + translation
    else:
        return None
    if up @ -point < _FLOOR_SMALLEST_CAMERA_HEIGHT_METRES:
        return None
    return source, up, point


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _world_frame(
    relative_rotations: list[np.ndarray], relative_translations: list[np.ndarray], floor: _FloorSighting | None
) -> tuple[np.ndarray, np.ndarray]:
    up = floor.normal if floor is not None else _CAMERA_UP
    center = _capture_center(relative_rotations, relative_translations)
    origin = center - ((center - floor.point) @ up) * up if floor is not None else center
    forward = _CAMERA_FORWARD - (_CAMERA_FORWARD @ up) * up
    if np.linalg.norm(forward) < 1e-6:
        raise CalibrationError("The first camera looks straight down at the floor; it must look across the room.")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    return np.column_stack((right, forward, up)), origin


def _capture_center(relative_rotations: list[np.ndarray], relative_translations: list[np.ndarray]) -> np.ndarray:
    fallback = _FALLBACK_CENTER_DISTANCE_METRES * _CAMERA_FORWARD
    midpoints = []
    for rotation, translation in zip(relative_rotations[1:], relative_translations[1:]):
        position = -rotation.T @ translation
        direction = rotation.T @ _CAMERA_FORWARD
        # Closest points of the two optical axes: s along the first camera's, u along this one's.
        cosine = float(_CAMERA_FORWARD @ direction)
        denominator = 1.0 - cosine * cosine
        if denominator < 1e-6:
            continue
        along_first = float((_CAMERA_FORWARD @ position) - cosine * (direction @ position)) / denominator
        along_other = float(cosine * (_CAMERA_FORWARD @ position) - (direction @ position)) / denominator
        if along_first <= 0.0 or along_other <= 0.0:
            continue
        midpoints.append((along_first * _CAMERA_FORWARD + position + along_other * direction) / 2.0)
    return np.mean(midpoints, axis=0) if midpoints else fallback
