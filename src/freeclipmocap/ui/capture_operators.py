from pathlib import Path

import bpy

from ..capture.capture_job import CaptureJob
from ..capture.landmark_cache import LandmarkCache, cache_path_for, find_current_cache
from .video_operators import take_video_paths

MODELS_FOLDER = Path(__file__).resolve().parent.parent / "models"
_TIMER_INTERVAL_SECONDS = 0.2


class CaptureState:
    def __init__(self) -> None:
        self.job: CaptureJob | None = None
        self.status_message = ""
        self.camera_label = ""

    @property
    def is_running(self) -> bool:
        return self.job is not None and self.job.is_running

    def set_job(self, job: CaptureJob | None) -> None:
        self.job = job

    def set_status_message(self, message: str) -> None:
        self.status_message = message

    def set_camera_label(self, label: str) -> None:
        self.camera_label = label


capture_state = CaptureState()


class CaptureOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.capture"
    bl_label = "Capture"
    bl_description = (
        "Track the body and hands in every frame of the video and cache the landmarks next to it. "
        "With several cameras, every camera's video is captured in turn"
    )

    _timer: bpy.types.Timer | None = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return not capture_state.is_running and bool(context.scene.freeclipmocap.video_path)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        properties = context.scene.freeclipmocap
        video_paths = take_video_paths(properties)
        for camera_number, video_path in enumerate(video_paths, start=1):
            if not video_path.is_file():
                what = f"Camera {camera_number}: video" if len(video_paths) > 1 else "Video"
                self.report({"ERROR"}, f"{what} not found: {video_path}")
                return {"CANCELLED"}
        settings = properties.capture_settings()
        self._camera_count = len(video_paths)
        self._messages: list[str] = []
        self._pending: list[tuple[int, Path]] = []
        for camera_number, video_path in enumerate(video_paths, start=1):
            existing_cache_path = find_current_cache(video_path, settings)
            if existing_cache_path is None:
                self._pending.append((camera_number, video_path))
                continue
            existing_cache = LandmarkCache.load(existing_cache_path)
            if camera_number == 1:
                fill_end_frame(properties, existing_cache)
            self._messages.append(self._camera_prefix(camera_number) + summarize(existing_cache, "Already captured"))
        if not self._pending:
            message = " ".join(self._messages)
            capture_state.set_status_message(message)
            self.report({"INFO"}, message)
            return {"FINISHED"}

        capture_state.set_status_message("")
        self._start_next(context)
        self._timer = context.window_manager.event_timer_add(_TIMER_INTERVAL_SECONDS, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        job = capture_state.job
        if job is None:
            self._finish(context)
            return {"CANCELLED"}
        if event.type == "TIMER":
            redraw_sidebars(context)
            if not job.is_running:
                succeeded = self._record_result(context, job)
                if succeeded and self._pending:
                    self._start_next(context)
                    return {"PASS_THROUGH"}
                self._finish(context)
                self._report_results(succeeded)
                return {"FINISHED"}
        return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context) -> None:
        if capture_state.job is not None:
            capture_state.job.cancel()
        self._finish(context)

    def _camera_prefix(self, camera_number: int) -> str:
        return f"Camera {camera_number}: " if self._camera_count > 1 else ""

    def _start_next(self, context: bpy.types.Context) -> None:
        camera_number, video_path = self._pending.pop(0)
        settings = context.scene.freeclipmocap.capture_settings()
        job = CaptureJob(video_path, settings, MODELS_FOLDER, cache_path_for(video_path, settings))
        job.set_camera_number(camera_number)
        capture_state.set_job(job)
        capture_state.set_camera_label(self._camera_prefix(camera_number))
        job.start()

    def _finish(self, context: bpy.types.Context) -> None:
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        capture_state.set_camera_label("")
        redraw_sidebars(context)

    def _record_result(self, context: bpy.types.Context, job: CaptureJob) -> bool:
        prefix = self._camera_prefix(job.camera_number)
        if job.cache is not None:
            if job.camera_number == 1:
                fill_end_frame(context.scene.freeclipmocap, job.cache)
            self._messages.append(f"{prefix}{summarize(job.cache, 'Captured')} Cache: {job.cache_path}")
            return True
        if job.was_cancelled:
            self._messages.append(f"{prefix}Capture cancelled.")
        else:
            self._messages.append(f"{prefix}Capture failed: {job.error_message}")
        return False

    def _report_results(self, succeeded: bool) -> None:
        message = " ".join(self._messages)
        if succeeded:
            self.report({"INFO"}, message)
        elif capture_state.job is not None and capture_state.job.was_cancelled:
            self.report({"WARNING"}, message)
        else:
            self.report({"ERROR"}, message)
        capture_state.set_status_message(message)


class CancelCaptureOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.cancel_capture"
    bl_label = "Cancel"
    bl_description = "Stop the running capture; nothing is cached"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return capture_state.is_running

    def execute(self, context: bpy.types.Context) -> set[str]:
        if capture_state.job is not None:
            capture_state.job.cancel()
        return {"FINISHED"}


def fill_end_frame(properties: bpy.types.PropertyGroup, cache: LandmarkCache) -> None:
    if properties.video_end_frame == 0:
        properties.video_end_frame = cache.frame_count


def summarize(cache: LandmarkCache, verb: str) -> str:
    frame_count = max(cache.frame_count, 1)
    body_percent = 100.0 * cache.pose.found.sum() / frame_count
    left_hand_percent = 100.0 * cache.left_hand.found.sum() / frame_count
    right_hand_percent = 100.0 * cache.right_hand.found.sum() / frame_count
    return (
        f"{verb} {cache.frame_count} frames: body {body_percent:.0f}%, left hand {left_hand_percent:.0f}%, "
        f"right hand {right_hand_percent:.0f}%."
    )


def redraw_sidebars(context: bpy.types.Context) -> None:
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def cancel_running_capture() -> None:
    if capture_state.job is not None:
        capture_state.job.cancel()
        capture_state.job.wait()
    capture_state.set_job(None)


_classes = (CaptureOperator, CancelCaptureOperator)


def register() -> None:
    for operator_class in _classes:
        bpy.utils.register_class(operator_class)


def unregister() -> None:
    cancel_running_capture()
    for operator_class in reversed(_classes):
        bpy.utils.unregister_class(operator_class)
