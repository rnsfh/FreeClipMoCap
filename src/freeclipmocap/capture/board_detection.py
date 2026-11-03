import threading
from pathlib import Path

import cv2
import numpy as np

from ..constants import calibration_board
from .board_sightings import BoardSightings
from .capture_progress import CaptureCancelled, CaptureProgress

_PRINT_DOTS_PER_MILLIMETRE = 300.0 / 25.4
_LABEL_HEIGHT_SHARE_OF_PAGE = 0.012
_FALLBACK_FRAME_RATE = 30.0


class BoardDetectionError(Exception):
    pass


def create_board(square_size_metres: float) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, calibration_board.ARUCO_DICTIONARY_NAME))
    return cv2.aruco.CharucoBoard(
        (calibration_board.SQUARES_ACROSS, calibration_board.SQUARES_DOWN),
        square_size_metres,
        square_size_metres * calibration_board.MARKER_SHARE_OF_SQUARE,
        dictionary,
    )


def board_corner_positions(square_size_metres: float) -> np.ndarray:
    return np.asarray(create_board(square_size_metres).getChessboardCorners(), dtype=np.float64).reshape(-1, 3)


def save_board_image(path: Path) -> None:
    page_width = int(round(calibration_board.A4_PAPER_WIDTH_MILLIMETRES * _PRINT_DOTS_PER_MILLIMETRE))
    page_height = int(round(calibration_board.A4_PAPER_HEIGHT_MILLIMETRES * _PRINT_DOTS_PER_MILLIMETRE))
    square_pixels = calibration_board.NOMINAL_SQUARE_MILLIMETRES * _PRINT_DOTS_PER_MILLIMETRE
    board_width = int(round(calibration_board.SQUARES_ACROSS * square_pixels))
    board_height = int(round(calibration_board.SQUARES_DOWN * square_pixels))
    board_image = create_board(calibration_board.NOMINAL_SQUARE_MILLIMETRES / 1000.0).generateImage(
        (board_width, board_height), marginSize=0
    )
    page = np.full((page_height, page_width), 255, dtype=np.uint8)
    left = (page_width - board_width) // 2
    top = (page_height - board_height) // 2
    page[top : top + board_height, left : left + board_width] = board_image
    label = (
        f"FreeClipMocap calibration board {calibration_board.SQUARES_ACROSS}x{calibration_board.SQUARES_DOWN}, "
        f"{calibration_board.ARUCO_DICTIONARY_NAME}, squares {calibration_board.NOMINAL_SQUARE_MILLIMETRES:g} mm "
        "at 100% scale. Measure one square after printing and enter its size."
    )
    font_scale = _LABEL_HEIGHT_SHARE_OF_PAGE * page_height / 22.0
    (label_width, _label_height), _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    label_origin = ((page_width - label_width) // 2, top + board_height + int(0.03 * page_height))
    cv2.putText(page, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), page):
        raise BoardDetectionError(f"Cannot write the board image to {path}.")


def detect_board(
    video_path: Path,
    first_frame: int,
    stop_frame: int,
    progress: CaptureProgress,
    cancel_event: threading.Event,
) -> BoardSightings:
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise BoardDetectionError(f"Cannot open the video {video_path}.")
    detector = cv2.aruco.CharucoDetector(create_board(calibration_board.NOMINAL_SQUARE_MILLIMETRES / 1000.0))
    frame_rate = video.get(cv2.CAP_PROP_FPS) or _FALLBACK_FRAME_RATE
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if stop_frame <= 0 or stop_frame > frame_count > 0:
        stop_frame = frame_count
    progress.set_expected_frame_count(max(stop_frame - first_frame, 0))

    frame_indices: list[int] = []
    timestamps_seconds: list[float] = []
    corner_points: list[np.ndarray] = []
    first_position_milliseconds: float | None = None
    width = 0
    height = 0
    try:
        frame_index = -1
        while True:
            if cancel_event.is_set():
                raise CaptureCancelled()
            frame_was_read, frame = video.read()
            if not frame_was_read:
                break
            frame_index += 1
            position_milliseconds = video.get(cv2.CAP_PROP_POS_MSEC)
            if first_position_milliseconds is None:
                first_position_milliseconds = position_milliseconds
            if frame_index < first_frame:
                continue
            if stop_frame > 0 and frame_index >= stop_frame:
                break
            height, width = frame.shape[:2]
            frame_indices.append(frame_index)
            timestamps_seconds.append((position_milliseconds - first_position_milliseconds) / 1000.0)
            corner_points.append(_corners_in_frame(detector, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
            progress.advance()
    finally:
        video.release()

    if not frame_indices:
        raise BoardDetectionError(f"No frames could be read from {video_path.name} in the chosen range.")
    return BoardSightings(
        video_file_name=video_path.name,
        image_width=width,
        image_height=height,
        frame_rate=frame_rate,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        timestamps_seconds=np.array(timestamps_seconds, dtype=np.float64),
        corner_points=np.stack(corner_points),
    )


def _corners_in_frame(detector: cv2.aruco.CharucoDetector, gray_frame: np.ndarray) -> np.ndarray:
    points = np.full((calibration_board.INNER_CORNER_COUNT, 2), np.nan, dtype=np.float32)
    corners, ids, _marker_corners, _marker_ids = detector.detectBoard(gray_frame)
    if ids is not None and len(ids):
        points[ids.ravel()] = corners.reshape(-1, 2)
    return points
