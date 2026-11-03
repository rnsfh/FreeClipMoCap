from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bl_ext.user_default.freeclipmocap.capture.board_detection import create_board
from bl_ext.user_default.freeclipmocap.constants import calibration_board

WIDTH = 1920
HEIGHT = 1080
FRAME_RATE = 30.0
SQUARE_SIZE_METRES = 0.048
_WORLD_UP = np.array([0.0, 0.0, 1.0])
_BOARD_PIXELS_PER_METRE = 3000.0
_LOOK_TARGET = np.array([0.0, 0.0, 0.7])


@dataclass(frozen=True, slots=True, eq=False)
class SyntheticCamera:
    position: np.ndarray
    focal_length_pixels: float
    rotation: np.ndarray
    translation: np.ndarray

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array([[self.focal_length_pixels, 0.0, WIDTH / 2.0], [0.0, self.focal_length_pixels, HEIGHT / 2.0], [0.0, 0.0, 1.0]])


def looking_camera(position: np.ndarray, focal_length_pixels: float) -> SyntheticCamera:
    forward = (_LOOK_TARGET - position) / np.linalg.norm(_LOOK_TARGET - position)
    right = np.cross(forward, _WORLD_UP)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    return SyntheticCamera(position, focal_length_pixels, rotation, -rotation @ position)


FRONT_CAMERA = looking_camera(np.array([0.0, -1.7, 1.6]), 1400.0)
SIDE_CAMERA = looking_camera(np.array([1.25, -1.15, 1.6]), 1550.0)


def board_facing(center: np.ndarray, yaw: float, pitch: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
    facing = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    viewer_direction = -facing
    right = np.cross(viewer_direction, _WORLD_UP)
    right /= np.linalg.norm(right)
    down = np.cross(viewer_direction, right)
    axes = np.column_stack((right, down, viewer_direction))
    return axes @ _rotation_about_x(pitch) @ _rotation_about_z(roll), center


def board_on_floor(center_on_floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.column_stack((np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), -_WORLD_UP)), center_on_floor


CLOSE_UP_SECONDS = 1.5
SHARED_SECONDS = 3.5
FLOOR_SHOT_SECONDS = 1.5
UPRIGHT_FROM_SECONDS = 2.0 * CLOSE_UP_SECONDS + SHARED_SECONDS
FLAT_FROM_SECONDS = UPRIGHT_FROM_SECONDS + FLOOR_SHOT_SECONDS
CALIBRATION_SECONDS = FLAT_FROM_SECONDS + FLOOR_SHOT_SECONDS
PAPER_HEIGHT_METRES = calibration_board.A4_PAPER_HEIGHT_MILLIMETRES / 1000.0
_CLOSE_UP_DISTANCE_METRES = 0.8


def calibration_board_pose(time_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    if time_seconds < CLOSE_UP_SECONDS:
        return board_close_to(FRONT_CAMERA, time_seconds)
    if time_seconds < 2.0 * CLOSE_UP_SECONDS:
        return board_close_to(SIDE_CAMERA, time_seconds)
    if time_seconds < UPRIGHT_FROM_SECONDS:
        return waving_board(time_seconds)
    if time_seconds < FLAT_FROM_SECONDS:
        return board_standing_on_floor(np.array([0.1, 0.25]))
    return board_on_floor(np.array([0.15, -0.1, 0.0]))


def board_standing_on_floor(floor_spot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    board_height = calibration_board.SQUARES_DOWN * SQUARE_SIZE_METRES
    paper_margin = (PAPER_HEIGHT_METRES - board_height) / 2.0
    return board_facing(np.array([floor_spot[0], floor_spot[1], paper_margin + board_height / 2.0]), np.radians(24.0), 0.0, 0.0)


def board_close_to(camera: SyntheticCamera, time_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    forward = camera.rotation[2]
    center = camera.position + _CLOSE_UP_DISTANCE_METRES * forward + 0.08 * np.array([
        np.sin(2.1 * time_seconds), 0.0, np.sin(2.9 * time_seconds)
    ])
    toward_camera = camera.position - center
    return board_facing(
        center,
        yaw=float(np.arctan2(toward_camera[0], -toward_camera[1])) + 0.5 * np.sin(3.3 * time_seconds),
        pitch=-0.45 + 0.5 * np.sin(2.6 * time_seconds + 0.5),
        roll=0.4 * np.sin(1.9 * time_seconds),
    )


def waving_board(time_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.array([
        0.25 * np.sin(1.3 * time_seconds),
        0.15 * np.sin(0.9 * time_seconds),
        0.95 + 0.2 * np.sin(1.7 * time_seconds),
    ])
    return board_facing(
        center,
        yaw=np.radians(24.0) + 0.35 * np.sin(1.1 * time_seconds),
        pitch=0.35 * np.sin(1.5 * time_seconds + 1.0),
        roll=0.3 * np.sin(0.7 * time_seconds),
    )


def write_board_video(
    path: Path,
    camera: SyntheticCamera,
    started_at_seconds: float,
    frame_count: int,
    board_pose_at: Callable[[float], tuple[np.ndarray, np.ndarray]],
) -> Path:
    board = create_board(SQUARE_SIZE_METRES)
    board_size_metres = np.array([calibration_board.SQUARES_ACROSS, calibration_board.SQUARES_DOWN]) * SQUARE_SIZE_METRES
    board_image = board.generateImage(tuple(int(size) for size in board_size_metres * _BOARD_PIXELS_PER_METRE), marginSize=0)
    blank = np.full_like(board_image, 255)
    metres_per_pixel = np.diag([1.0 / _BOARD_PIXELS_PER_METRE, 1.0 / _BOARD_PIXELS_PER_METRE, 1.0])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, (WIDTH, HEIGHT))
    for frame_index in range(frame_count):
        axes, center = board_pose_at(started_at_seconds + frame_index / FRAME_RATE)
        corner_origin = center - axes @ np.array([board_size_metres[0] / 2.0, board_size_metres[1] / 2.0, 0.0])
        homography = camera.camera_matrix @ np.column_stack((
            camera.rotation @ axes[:, 0], camera.rotation @ axes[:, 1], camera.rotation @ corner_origin + camera.translation
        )) @ metres_per_pixel
        frame = np.full((HEIGHT, WIDTH), 110, dtype=np.uint8)
        warped = cv2.warpPerspective(board_image, homography, (WIDTH, HEIGHT), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(blank, homography, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
        frame[mask > 0] = warped[mask > 0]
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def _rotation_about_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotation_about_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
