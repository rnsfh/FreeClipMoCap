from pathlib import Path

import bpy

from ..capture.capture_job import CaptureJob
from ..capture.landmark_cache import LandmarkCache, cache_path_for, find_current_cache

MODELS_FOLDER = Path(__file__).resolve().parent.parent / "models"
_TIMER_INTERVAL_SECONDS = 0.2


class CaptureState:
    def __init__(self) -> None:
        self.job: CaptureJob | None = None
        self.status_message = ""

    @property
    def is_running(self) -> bool:
        return self.job is not None and self.job.is_running

    def set_job(self, job: CaptureJob | None) -> None:
        self.job = job

    def set_status_message(self, message: str) -> None:
        self.status_message = message


capture_state = CaptureState()


class CaptureOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.capture"
    bl_label = "Capture"
    bl_description = "Track the body and hands in every frame of the video and cache the landmarks next to it"

    _timer: bpy.types.Timer | None = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return not capture_state.is_running and bool(context.scene.freeclipmocap.video_path)

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        properties = context.scene.freeclipmocap
        video_path = Path(bpy.path.abspath(properties.video_path))
        if not video_path.is_file():
            self.report({"ERROR"}, f"Video not found: {video_path}")
            return {"CANCELLED"}
        settings = properties.capture_settings()
        existing_cache_path = find_current_cache(video_path, settings)
        if existing_cache_path is not None:
            existing_cache = LandmarkCache.load(existing_cache_path)
            fill_end_frame(properties, existing_cache)
            message = summarize(existing_cache, "Already captured")
            capture_state.set_status_message(message)
            self.report({"INFO"}, message)
            return {"FINISHED"}

        job = CaptureJob(video_path, settings, MODELS_FOLDER, cache_path_for(video_path, settings))
        capture_state.set_job(job)
        capture_state.set_status_message("")
        job.start()
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
                self._finish(context)
                self._report_result(context, job)
                return {"FINISHED"}
        return {"PASS_THROUGH"}

    def cancel(self, context: bpy.types.Context) -> None:
        if capture_state.job is not None:
            capture_state.job.cancel()
        self._finish(context)

    def _finish(self, context: bpy.types.Context) -> None:
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        redraw_sidebars(context)

    def _report_result(self, context: bpy.types.Context, job: CaptureJob) -> None:
        if job.cache is not None:
            fill_end_frame(context.scene.freeclipmocap, job.cache)
            message = summarize(job.cache, "Captured")
            self.report({"INFO"}, f"{message} Cache: {job.cache_path}")
        elif job.was_cancelled:
            message = "Capture cancelled."
            self.report({"WARNING"}, message)
        else:
            message = f"Capture failed: {job.error_message}"
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
