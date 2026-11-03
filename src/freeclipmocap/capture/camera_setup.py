import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from ..constants.camera_lens import FULL_FRAME_WIDTH_MILLIMETRES
from .capture_settings import CaptureSettings
from .saved_lens import SavedLens

_FORMAT_VERSION = 2
_FOLDER_NAME = "FreeClipMocap"
_FILE_SUFFIX = ".calibration.json"


class CameraSetupFormatError(Exception):
    pass


class FloorSource(StrEnum):
    NONE = "none"
    BOARD_UPRIGHT = "board-upright"
    BOARD_FLAT = "board-flat"
    FEET = "feet"


@dataclass(frozen=True, slots=True)
class CalibrationVideo:
    file_name: str
    file_size: int
    modified_time_nanoseconds: int

    @classmethod
    def of(cls, video_path: Path) -> "CalibrationVideo":
        status = video_path.stat()
        return cls(video_path.name, status.st_size, status.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class TypedLens:
    focal_length_millimetres: float
    keep_fixed: bool

    def focal_length_pixels(self, image_width: int, image_height: int) -> float:
        return self.focal_length_millimetres * max(image_width, image_height) / FULL_FRAME_WIDTH_MILLIMETRES


@dataclass(frozen=True, slots=True)
class BodyCalibrationInputs:
    cameras_apart_metres: float
    capture_settings: CaptureSettings


@dataclass(frozen=True, slots=True)
class CalibrationInputs:
    videos: tuple[CalibrationVideo, ...]
    start_delays_seconds: tuple[float, ...]
    saved_lenses: tuple[SavedLens | None, ...]
    typed_lenses: tuple[TypedLens | None, ...]
    square_size_metres: float
    edge_below_squares_metres: float
    first_frame: int
    stop_frame: int
    body: BodyCalibrationInputs | None = None

    @property
    def uses_body(self) -> bool:
        return self.body is not None

    def to_values(self) -> dict[str, object]:
        return {
            "videos": [
                {"file_name": video.file_name, "file_size": video.file_size, "modified_time_nanoseconds": video.modified_time_nanoseconds}
                for video in self.videos
            ],
            "start_delays_seconds": list(self.start_delays_seconds),
            "saved_lenses": [None if lens is None else lens.to_values() for lens in self.saved_lenses],
            "typed_lenses": [
                None if lens is None else {"focal_length_millimetres": lens.focal_length_millimetres, "keep_fixed": lens.keep_fixed}
                for lens in self.typed_lenses
            ],
            "square_size_metres": self.square_size_metres,
            "edge_below_squares_metres": self.edge_below_squares_metres,
            "first_frame": self.first_frame,
            "stop_frame": self.stop_frame,
            "body": None
            if self.body is None
            else {"cameras_apart_metres": self.body.cameras_apart_metres, "capture_settings": self.body.capture_settings.to_json()},
        }

    @classmethod
    def from_values(cls, values: dict[str, object]) -> "CalibrationInputs":
        return cls(
            videos=tuple(
                CalibrationVideo(str(video["file_name"]), int(video["file_size"]), int(video["modified_time_nanoseconds"]))
                for video in values["videos"]
            ),
            start_delays_seconds=tuple(float(delay) for delay in values["start_delays_seconds"]),
            saved_lenses=tuple(None if lens is None else SavedLens.from_values(lens) for lens in values["saved_lenses"]),
            typed_lenses=tuple(
                None if lens is None else TypedLens(float(lens["focal_length_millimetres"]), bool(lens["keep_fixed"]))
                for lens in values["typed_lenses"]
            ),
            square_size_metres=float(values["square_size_metres"]),
            edge_below_squares_metres=float(values["edge_below_squares_metres"]),
            first_frame=int(values["first_frame"]),
            stop_frame=int(values["stop_frame"]),
            body=None
            if values["body"] is None
            else BodyCalibrationInputs(
                float(values["body"]["cameras_apart_metres"]), CaptureSettings.from_json(str(values["body"]["capture_settings"]))
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class CalibratedCamera:
    video_file_name: str
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    lens_error_pixels: float
    picture_delay_correction_seconds: float

    @property
    def position(self) -> np.ndarray:
        return -self.rotation.T @ self.translation

    @property
    def focal_length_millimetres(self) -> float:
        return float(self.camera_matrix[0, 0]) * FULL_FRAME_WIDTH_MILLIMETRES / max(self.image_width, self.image_height)

    def project(self, world_points: np.ndarray) -> np.ndarray:
        camera_points = world_points @ self.rotation.T + self.translation
        normalized = camera_points[:, :2] / camera_points[:, 2:3]
        squared_radius = np.sum(np.square(normalized), axis=1, keepdims=True)
        radial_one, radial_two, tangential_one, tangential_two, radial_three = self.distortion
        radial = 1.0 + squared_radius * (radial_one + squared_radius * (radial_two + squared_radius * radial_three))
        x, y = normalized[:, 0:1], normalized[:, 1:2]
        distorted = np.hstack((
            x * radial + 2.0 * tangential_one * x * y + tangential_two * (squared_radius + 2.0 * x * x),
            y * radial + tangential_one * (squared_radius + 2.0 * y * y) + 2.0 * tangential_two * x * y,
        ))
        return distorted @ self.camera_matrix[:2, :2].T + self.camera_matrix[:2, 2]


@dataclass(frozen=True, slots=True, eq=False)
class CameraSetup:
    inputs: CalibrationInputs
    cameras: tuple[CalibratedCamera, ...]
    floor_source: FloorSource
    floor_camera_index: int | None
    floor_video_frames: tuple[int, int] | None
    pair_error_pixels: tuple[float, ...]

    @property
    def floor_found(self) -> bool:
        return self.floor_source != FloorSource.NONE

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "format_version": _FORMAT_VERSION,
            "inputs": self.inputs.to_values(),
            "floor_source": str(self.floor_source),
            "floor_camera_index": self.floor_camera_index,
            "floor_video_frames": list(self.floor_video_frames) if self.floor_video_frames else None,
            "pair_error_pixels": list(self.pair_error_pixels),
            "cameras": [_camera_values(camera) for camera in self.cameras],
        }
        temporary_path = path.with_name(path.name + ".writing")
        temporary_path.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: Path) -> "CameraSetup":
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            format_version = int(values["format_version"])
            if format_version != _FORMAT_VERSION:
                raise CameraSetupFormatError(
                    f"{path.name} has format version {format_version}, expected {_FORMAT_VERSION}."
                )
            floor_frames = values["floor_video_frames"]
            return cls(
                inputs=CalibrationInputs.from_values(values["inputs"]),
                cameras=tuple(_camera_from_values(camera_values) for camera_values in values["cameras"]),
                floor_source=FloorSource(values["floor_source"]),
                floor_camera_index=None if values["floor_camera_index"] is None else int(values["floor_camera_index"]),
                floor_video_frames=None if floor_frames is None else (int(floor_frames[0]), int(floor_frames[1])),
                pair_error_pixels=tuple(float(error) for error in values["pair_error_pixels"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CameraSetupFormatError(f"{path.name} is not a valid calibration file: {error}") from error


def calibration_path_for(reference_video_path: Path) -> Path:
    return reference_video_path.parent / _FOLDER_NAME / f"{reference_video_path.name}{_FILE_SUFFIX}"


def find_current_setup(path: Path, inputs: CalibrationInputs) -> CameraSetup | None:
    if not path.is_file():
        return None
    try:
        setup = CameraSetup.load(path)
    except (OSError, CameraSetupFormatError):
        return None
    return setup if setup.inputs == inputs else None


def _camera_values(camera: CalibratedCamera) -> dict[str, object]:
    return {
        "video_file_name": camera.video_file_name,
        "image_width": camera.image_width,
        "image_height": camera.image_height,
        "camera_matrix": camera.camera_matrix.tolist(),
        "distortion": camera.distortion.tolist(),
        "rotation": camera.rotation.tolist(),
        "translation": camera.translation.tolist(),
        "lens_error_pixels": camera.lens_error_pixels,
        "picture_delay_correction_seconds": camera.picture_delay_correction_seconds,
    }


def _camera_from_values(values: dict[str, object]) -> CalibratedCamera:
    camera = CalibratedCamera(
        video_file_name=str(values["video_file_name"]),
        image_width=int(values["image_width"]),
        image_height=int(values["image_height"]),
        camera_matrix=np.array(values["camera_matrix"], dtype=np.float64).reshape(3, 3),
        distortion=np.array(values["distortion"], dtype=np.float64).reshape(5),
        rotation=np.array(values["rotation"], dtype=np.float64).reshape(3, 3),
        translation=np.array(values["translation"], dtype=np.float64).reshape(3),
        lens_error_pixels=float(values["lens_error_pixels"]),
        picture_delay_correction_seconds=float(values["picture_delay_correction_seconds"]),
    )
    for array in (camera.camera_matrix, camera.distortion, camera.rotation, camera.translation):
        array.flags.writeable = False
    return camera
