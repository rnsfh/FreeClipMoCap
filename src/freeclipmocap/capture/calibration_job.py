import math
import threading
from collections.abc import Sequence
from pathlib import Path

from .camera_setup import CalibrationInputs, CameraSetup
from .capture_progress import CaptureCancelled, CaptureProgress
from .landmark_cache import LandmarkCache, cache_path_for, find_current_cache

_RANGE_MARGIN_SECONDS = 1.0


class CalibrationProgress:
    def __init__(self, camera_count: int) -> None:
        self.camera_count = camera_count
        self.camera_index = 0
        self.current = CaptureProgress()

    def start_camera(self, camera_index: int) -> CaptureProgress:
        self.camera_index = camera_index
        self.current = CaptureProgress()
        return self.current

    @property
    def fraction(self) -> float:
        return min((self.camera_index + self.current.fraction) / max(self.camera_count, 1), 1.0)


class CalibrationJob:
    def __init__(
        self, video_paths: Sequence[Path], inputs: CalibrationInputs, setup_path: Path, models_folder: Path | None = None
    ) -> None:
        self.video_paths = list(video_paths)
        self.inputs = inputs
        self.setup_path = setup_path
        self.models_folder = models_folder
        self.progress = CalibrationProgress(len(self.video_paths))
        self.setup: CameraSetup | None = None
        self.error_message: str | None = None
        self.was_cancelled = False
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="FreeClipMocap calibration", daemon=True)

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()

    def wait(self, timeout_seconds: float | None = None) -> None:
        self._thread.join(timeout_seconds)

    def _run(self) -> None:
        try:
            setup = self._calibrate_from_bodies() if self.inputs.uses_body else self._calibrate_from_the_board()
            setup.save(self.setup_path)
            self.setup = setup
        except CaptureCancelled:
            self.was_cancelled = True
        except Exception as error:
            self.error_message = f"{type(error).__name__}: {error}"

    def _calibrate_from_the_board(self) -> CameraSetup:
        from .board_detection import detect_board, video_frame_rate
        from .camera_calibration import calibrate_cameras

        reference_rate = video_frame_rate(self.video_paths[0])
        sightings = []
        for camera_index, video_path in enumerate(self.video_paths):
            first_frame, stop_frame = self._frame_range(camera_index, reference_rate, video_frame_rate(video_path))
            progress = self.progress.start_camera(camera_index)
            sightings.append(detect_board(video_path, first_frame, stop_frame, progress, self._cancel_event))
        return calibrate_cameras(sightings, self.inputs)

    def _calibrate_from_bodies(self) -> CameraSetup:
        from .body_calibration import calibrate_cameras_from_bodies

        caches = [self._landmarks(camera_index, video_path) for camera_index, video_path in enumerate(self.video_paths)]
        return calibrate_cameras_from_bodies(caches, [video_path.name for video_path in self.video_paths], self.inputs)

    def _landmarks(self, camera_index: int, video_path: Path) -> LandmarkCache:
        settings = self.inputs.body.capture_settings
        progress = self.progress.start_camera(camera_index)
        cache_path = find_current_cache(video_path, settings)
        if cache_path is not None:
            progress.set_expected_frame_count(1)
            progress.advance()
            return LandmarkCache.load(cache_path)
        if self.models_folder is None:
            raise ValueError("The tracking models' folder is needed to capture the body.")
        from .landmark_capture import capture_landmarks

        cache = capture_landmarks(video_path, settings, self.models_folder, progress, self._cancel_event)
        cache.save(cache_path_for(video_path, settings))
        return cache

    def _frame_range(self, camera_index: int, reference_rate: float, frame_rate: float) -> tuple[int, int]:
        if camera_index == 0:
            return self.inputs.first_frame, self.inputs.stop_frame
        delay = self.inputs.start_delays_seconds[camera_index]
        first_seconds = self.inputs.first_frame / reference_rate - delay - _RANGE_MARGIN_SECONDS
        first_frame = max(math.floor(first_seconds * frame_rate), 0)
        if self.inputs.stop_frame <= 0:
            return first_frame, 0
        stop_seconds = self.inputs.stop_frame / reference_rate - delay + _RANGE_MARGIN_SECONDS
        return first_frame, max(math.ceil(stop_seconds * frame_rate), first_frame + 1)
