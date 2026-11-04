from dataclasses import replace

import numpy as np

from ..capture.landmark_cache import LandmarkCache
from ..constants import pose_landmarks

_SMALLEST_VISIBILITY = 0.5
_FEWEST_FITTING_POINTS = 4
_LENGTH_PERCENTILE = 75.0
_LEG_PARENTS = {
    pose_landmarks.LEFT_KNEE: pose_landmarks.LEFT_HIP,
    pose_landmarks.LEFT_ANKLE: pose_landmarks.LEFT_KNEE,
    pose_landmarks.LEFT_HEEL: pose_landmarks.LEFT_ANKLE,
    pose_landmarks.LEFT_FOOT_INDEX: pose_landmarks.LEFT_ANKLE,
    pose_landmarks.RIGHT_KNEE: pose_landmarks.RIGHT_HIP,
    pose_landmarks.RIGHT_ANKLE: pose_landmarks.RIGHT_KNEE,
    pose_landmarks.RIGHT_HEEL: pose_landmarks.RIGHT_ANKLE,
    pose_landmarks.RIGHT_FOOT_INDEX: pose_landmarks.RIGHT_ANKLE,
}
_HIPS = (pose_landmarks.LEFT_HIP, pose_landmarks.RIGHT_HIP)
_MIRRORED = {
    pose_landmarks.LEFT_KNEE: pose_landmarks.RIGHT_KNEE,
    pose_landmarks.LEFT_ANKLE: pose_landmarks.RIGHT_ANKLE,
    pose_landmarks.LEFT_HEEL: pose_landmarks.RIGHT_HEEL,
    pose_landmarks.LEFT_FOOT_INDEX: pose_landmarks.RIGHT_FOOT_INDEX,
}


def with_legs_from_the_video(cache: LandmarkCache, focal_length_pixels: float) -> LandmarkCache:
    # MediaPipe's one-camera 3D folds straight legs toward or away from the lens. Each leg joint goes back on the
    # camera ray through its 2D point, at its bone's length from its parent joint; MediaPipe only chooses whether the
    # bone points toward or away from the camera (the two places where the ray meets that sphere).
    pixels = cache.pose.image_landmarks[..., :2].astype(np.float64) * (cache.video.width, cache.video.height)
    centred = (pixels - (cache.video.width / 2.0, cache.video.height / 2.0)) / focal_length_pixels
    tracked = cache.pose.world_landmarks.astype(np.float64)
    visible = cache.pose.found[:, None] & (np.nan_to_num(cache.pose.visibility) >= _SMALLEST_VISIBILITY)
    visible &= np.isfinite(centred[..., 0]) & np.isfinite(tracked[..., 0])
    body_centre = _body_positions(centred, tracked, visible)
    usable = np.isfinite(body_centre[:, 0])
    if not usable.any():
        return cache
    rays = np.concatenate((centred, np.ones(centred.shape[:-1] + (1,))), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    lengths = _bone_lengths(centred[usable] * body_centre[usable, 2:3, None])
    rebuilt = _rebuilt_legs(tracked, rays, body_centre, lengths)

    world = tracked.copy()
    leg_points = list(_HIPS) + list(_LEG_PARENTS)
    rebuilt_middle = (rebuilt[:, _HIPS[0]] + rebuilt[:, _HIPS[1]]) / 2.0
    repaired = rebuilt[:, leg_points] - rebuilt_middle[:, None, :]
    keep = usable[:, None] & np.isfinite(repaired).all(axis=-1)
    world[:, leg_points] = np.where(keep[..., None], repaired, tracked[:, leg_points])
    return replace(cache, pose=replace(cache.pose, world_landmarks=world.astype(np.float32)))


def _rebuilt_legs(
    tracked: np.ndarray, rays: np.ndarray, body_centre: np.ndarray, lengths: dict[int | str, float]
) -> np.ndarray:
    rebuilt = tracked + body_centre[:, None, :]
    hips_middle = (tracked[:, _HIPS[0]] + tracked[:, _HIPS[1]]) / 2.0
    for hip in _HIPS:
        rebuilt[:, hip] = _on_ray(body_centre, tracked[:, hip] - hips_middle, rays[:, hip], lengths["hip"])
    for joint, parent in _LEG_PARENTS.items():
        rebuilt[:, joint] = _on_ray(rebuilt[:, parent], tracked[:, joint] - tracked[:, parent], rays[:, joint], lengths[joint])
    return rebuilt


def _body_positions(centred: np.ndarray, tracked: np.ndarray, visible: np.ndarray) -> np.ndarray:
    # Where MediaPipe's hip-centred skeleton sits in front of the lens: the shift t that best puts each point on its
    # camera ray, (x + t) - u (z + t_z) = 0 for x and y, a linear least-squares problem per frame.
    weight = visible.astype(np.float64)
    rows_x = np.stack((np.ones_like(weight), np.zeros_like(weight), -np.nan_to_num(centred[..., 0])), axis=-1)
    rows_y = np.stack((np.zeros_like(weight), np.ones_like(weight), -np.nan_to_num(centred[..., 1])), axis=-1)
    values_x = np.nan_to_num(centred[..., 0] * tracked[..., 2] - tracked[..., 0])
    values_y = np.nan_to_num(centred[..., 1] * tracked[..., 2] - tracked[..., 1])
    normal = np.einsum("fp,fpi,fpj->fij", weight, rows_x, rows_x) + np.einsum("fp,fpi,fpj->fij", weight, rows_y, rows_y)
    right = np.einsum("fp,fpi,fp->fi", weight, rows_x, values_x) + np.einsum("fp,fpi,fp->fi", weight, rows_y, values_y)
    positions = np.full((len(centred), 3), np.nan)
    enough = visible.sum(axis=1) >= _FEWEST_FITTING_POINTS
    positions[enough] = np.linalg.solve(normal[enough], right[enough][..., None])[..., 0]
    positions[positions[:, 2] <= 0.0] = np.nan
    return positions


def _bone_lengths(in_plane: np.ndarray) -> dict[int | str, float]:
    # A bone never looks longer than it is, so a high percentile of its length laid in the body's depth plane is its
    # length in the same scale as that depth. The 75th percentile matched the user's two-camera take best (shins
    # within 2-5°); higher ones let tracking jitter lengthen the bones, and lengthened bones bend in depth.
    def length(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.nanpercentile(np.linalg.norm(first - second, axis=-1), _LENGTH_PERCENTILE))

    lengths: dict[int | str, float] = {joint: length(in_plane[:, joint], in_plane[:, parent]) for joint, parent in _LEG_PARENTS.items()}
    for left, right in _MIRRORED.items():
        lengths[left] = lengths[right] = max(lengths[left], lengths[right])
    lengths["hip"] = length(in_plane[:, _HIPS[0]], in_plane[:, _HIPS[1]]) / 2.0
    return lengths


def _on_ray(parents: np.ndarray, tracked_offsets: np.ndarray, rays: np.ndarray, length: float) -> np.ndarray:
    along = np.sum(rays * parents, axis=-1)
    discriminant = along**2 - np.sum(parents * parents, axis=-1) + length**2
    root = np.sqrt(np.maximum(discriminant, 0.0))
    nearer = rays * (along - root)[:, None]
    farther = rays * (along + root)[:, None]
    guess = parents + tracked_offsets
    choose_farther = np.linalg.norm(farther - guess, axis=-1) < np.linalg.norm(nearer - guess, axis=-1)
    on_ray = np.where(choose_farther[:, None], farther, nearer)
    # A ray that misses the bone's reach: the bone points at the ray's nearest point, at its own length.
    nearest = rays * along[:, None]
    towards = nearest - parents
    with np.errstate(invalid="ignore", divide="ignore"):
        missed = parents + length * towards / np.linalg.norm(towards, axis=-1, keepdims=True)
    points = np.where((discriminant >= 0.0)[:, None], on_ray, missed)
    unseen = ~np.isfinite(rays[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        fallback = parents + length * tracked_offsets / np.linalg.norm(tracked_offsets, axis=-1, keepdims=True)
    return np.where(unseen[:, None], fallback, points)
