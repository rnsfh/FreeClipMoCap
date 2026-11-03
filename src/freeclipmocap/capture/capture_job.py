import threading
from pathlib import Path

from .capture_progress import CaptureCancelled, CaptureProgress
from .capture_settings import CaptureSettings
from .landmark_cache import LandmarkCache


class CaptureJob:
    def __init__(self, video_path: Path, settings: CaptureSettings, models_folder: Path, cache_path: Path) -> None:
        self.video_path = video_path
        self.settings = settings
        self.models_folder = models_folder
        self.cache_path = cache_path
        self.progress = CaptureProgress()
        self.cache: LandmarkCache | None = None
        self.error_message: str | None = None
        self.was_cancelled = False
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="FreeClipMocap capture", daemon=True)

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
            from .landmark_capture import capture_landmarks

            cache = capture_landmarks(
                self.video_path, self.settings, self.models_folder, self.progress, self._cancel_event
            )
            cache.save(self.cache_path)
            self.cache = cache
        except CaptureCancelled:
            self.was_cancelled = True
        except Exception as error:
            self.error_message = f"{type(error).__name__}: {error}"
