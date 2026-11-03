from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .rotations import normalized

_MINIMUM_POINTS = 12
_START_DIRECTIONS = 8
_ROBUST_SHARE_OF_IMAGE_HEIGHT = 0.02
_NEAREST_DEPTH = 0.01
_MAXIMUM_ITERATIONS = 100
_LARGEST_DAMPING = 1e10
_DIFFERENCE_STEP = 1e-6
_CONVERGED_COST_SHARE = 1e-9


class CameraFitError(Exception):
    pass


@dataclass(frozen=True, slots=True, eq=False)
class FittedCamera:
    rotation: np.ndarray
    position: np.ndarray
    median_error_pixels: float


def project(
    world_points: np.ndarray,
    rotation: np.ndarray,
    position: np.ndarray,
    focal_length_pixels: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    camera_points = (world_points - position) @ rotation
    depths = np.maximum(-camera_points[:, 2], _NEAREST_DEPTH)
    image_center = np.array([image_width / 2.0, image_height / 2.0])
    return image_center + focal_length_pixels * np.stack((camera_points[:, 0], -camera_points[:, 1]), axis=-1) / depths[:, None]


def fit_camera(
    world_points: np.ndarray,
    image_points: np.ndarray,
    image_width: int,
    image_height: int,
    focal_length_pixels: float,
    up: np.ndarray,
) -> FittedCamera:
    if len(world_points) < _MINIMUM_POINTS:
        raise CameraFitError(
            f"Only {len(world_points)} tracked joints could be paired with the rig; at least {_MINIMUM_POINTS} are needed."
        )
    fits = [
        _fit_from_start(start_rotation, start_position, focal_length_pixels, world_points, image_points, image_width, image_height)
        for start_rotation, start_position in _starts(world_points, image_points, focal_length_pixels, up)
    ]
    _cost, rotation, position = min(fits, key=lambda fit: fit[0])
    errors = np.linalg.norm(
        project(world_points, rotation, position, focal_length_pixels, image_width, image_height) - image_points, axis=1
    )
    return FittedCamera(rotation, position, float(np.median(errors)))


def _fit_from_start(
    start_rotation: np.ndarray,
    start_position: np.ndarray,
    focal_length_pixels: float,
    world_points: np.ndarray,
    image_points: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    robust_scale = _ROBUST_SHARE_OF_IMAGE_HEIGHT * image_height

    def weighted_errors(parameters: np.ndarray) -> np.ndarray:
        rotation, position = _camera(parameters, start_rotation)
        errors = project(world_points, rotation, position, focal_length_pixels, image_width, image_height) - image_points
        lengths = np.linalg.norm(errors, axis=1)
        # Huber-like: errors beyond robust_scale count linearly, so mistracked joints pull the camera less.
        weights = np.sqrt(np.minimum(1.0, robust_scale / np.maximum(lengths, 1e-12)))
        return (errors * weights[:, None]).ravel()

    parameters, cost = _least_squares(weighted_errors, np.concatenate((np.zeros(3), start_position)))
    return (cost, *_camera(parameters, start_rotation))


def _starts(
    world_points: np.ndarray, image_points: np.ndarray, focal_length_pixels: float, up: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    up = normalized(up)
    target = np.median(world_points, axis=0)
    heights = world_points @ up
    world_span = max(float(np.percentile(heights, 95) - np.percentile(heights, 5)), 1e-3)
    image_span = max(float(np.percentile(image_points[:, 1], 95) - np.percentile(image_points[:, 1], 5)), 1.0)
    distance = focal_length_pixels * world_span / image_span
    helper = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first_horizontal = normalized(np.cross(up, helper))
    second_horizontal = np.cross(up, first_horizontal)
    starts = []
    for angle in np.linspace(0.0, 2.0 * np.pi, _START_DIRECTIONS, endpoint=False):
        back = np.cos(angle) * first_horizontal + np.sin(angle) * second_horizontal
        right = normalized(np.cross(up, back))
        rotation = np.column_stack((right, np.cross(back, right), back))
        starts.append((rotation, target + distance * back))
    return starts


def _camera(parameters: np.ndarray, start_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _rotation_from_vector(parameters[:3]) @ start_rotation, parameters[3:6]


def _rotation_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = rotation_vector / angle
    cross_matrix = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross_matrix + (1.0 - np.cos(angle)) * cross_matrix @ cross_matrix


def _least_squares(residuals: Callable[[np.ndarray], np.ndarray], start: np.ndarray) -> tuple[np.ndarray, float]:
    parameters = start
    current = residuals(parameters)
    cost = float(current @ current)
    damping = 1e-3
    for _ in range(_MAXIMUM_ITERATIONS):
        jacobian = _numeric_jacobian(residuals, parameters, current)
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        improvement = 0.0
        while damping < _LARGEST_DAMPING:
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            candidate = parameters + step
            candidate_residuals = residuals(candidate)
            candidate_cost = float(candidate_residuals @ candidate_residuals)
            if candidate_cost < cost:
                improvement = cost - candidate_cost
                parameters, current, cost = candidate, candidate_residuals, candidate_cost
                damping = max(damping / 10.0, 1e-12)
                break
            damping *= 10.0
        if improvement <= _CONVERGED_COST_SHARE * max(cost, 1e-12):
            break
    return parameters, cost


def _numeric_jacobian(
    residuals: Callable[[np.ndarray], np.ndarray], parameters: np.ndarray, current: np.ndarray
) -> np.ndarray:
    jacobian = np.empty((len(current), len(parameters)))
    for index in range(len(parameters)):
        step = _DIFFERENCE_STEP * max(1.0, abs(parameters[index]))
        moved = parameters.copy()
        moved[index] += step
        jacobian[:, index] = (residuals(moved) - current) / step
    return jacobian
