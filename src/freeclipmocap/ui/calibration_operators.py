from collections.abc import Sequence
from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper

from ..capture.calibration_job import CalibrationJob
from ..capture.camera_setup import (
    BodyCalibrationInputs,
    CalibratedCamera,
    CalibrationInputs,
    CalibrationVideo,
    CameraSetup,
    FloorSource,
    TypedLens,
    calibration_path_for,
    find_current_setup,
)
from ..capture.saved_lens import SavedLens, SavedLensError, delete_lens, is_valid_lens_name, lens_path
from ..solver.audio_sync import AudioSync, AudioSyncError, find_start_delay
from .capture_operators import MODELS_FOLDER, capture_state, redraw_sidebars
from .lens_folder import find_lens, user_lens_folder
from .phone_cameras import place_phone_cameras
from .video_operators import load_take_clips, take_start_delays_seconds, take_video_paths
from .video_audio import VideoAudioError, read_audio

_TIMER_INTERVAL_SECONDS = 0.2
_LARGEST_SYNC_DELAY_SECONDS = 30.0
_CLEAR_PEAK_RATIO = 1.3
_CALIBRATE_FIRST = "Calibrate first; the lens is saved from the last calibration of these videos."
_FLOOR_DESCRIPTIONS = {
    FloorSource.BOARD_UPRIGHT: "the board standing on it",
    FloorSource.BOARD_FLAT: "the board lying on it",
    FloorSource.FEET: "the feet",
}


class CalibrationRequestError(Exception):
    pass


class CalibrationState:
    def __init__(self) -> None:
        self.job: CalibrationJob | None = None
        self.setup: CameraSetup | None = None
        self.status_message = ""

    @property
    def is_running(self) -> bool:
        return self.job is not None and self.job.is_running

    def set_job(self, job: CalibrationJob | None) -> None:
        self.job = job

    def set_setup(self, setup: CameraSetup | None) -> None:
        self.setup = setup

    def set_status_message(self, message: str) -> None:
        self.status_message = message


calibration_state = CalibrationState()


def camera_settings(properties: bpy.types.PropertyGroup, camera_index: int) -> bpy.types.PropertyGroup:
    return properties if camera_index == 0 else properties.extra_videos[camera_index - 1]


def chosen_calibration_paths(properties: bpy.types.PropertyGroup) -> list[str]:
    if properties.use_separate_calibration_videos:
        return [properties.calibration_video_path] + [camera.calibration_video_path for camera in properties.extra_videos]
    return [properties.video_path] + [camera.video_path for camera in properties.extra_videos]


def chosen_saved_lenses(properties: bpy.types.PropertyGroup) -> list[SavedLens | None]:
    saved_lenses = []
    for camera_index in range(len(properties.extra_videos) + 1):
        settings = camera_settings(properties, camera_index)
        lens_name = "" if settings.calibration_lens_is_typed else settings.calibration_lens_name
        saved_lens = find_lens(lens_name) if lens_name else None
        if lens_name and saved_lens is None:
            raise CalibrationRequestError(
                f"Camera {camera_index + 1}: the saved lens {lens_name} is not in your FreeClipMocap user folder. "
                "Pick another lens or Measure."
            )
        saved_lenses.append(saved_lens)
    return saved_lenses


def uses_body(properties: bpy.types.PropertyGroup) -> bool:
    return properties.calibration_method == "BODY"


def lens_is_missing_for_body(settings: bpy.types.PropertyGroup) -> bool:
    return not settings.calibration_lens_is_typed and not settings.calibration_lens_name


def chosen_typed_lenses(properties: bpy.types.PropertyGroup) -> list[TypedLens | None]:
    typed_lenses = []
    for camera_index in range(len(properties.extra_videos) + 1):
        settings = camera_settings(properties, camera_index)
        typed_lenses.append(
            TypedLens(settings.calibration_focal_length_millimetres, settings.calibration_keep_focal_length)
            if settings.calibration_lens_is_typed
            else None
        )
    return typed_lenses


def calibrated_camera(context: bpy.types.Context, camera_index: int) -> CalibratedCamera | None:
    setup = calibration_state.setup
    paths = chosen_calibration_paths(context.scene.freeclipmocap)
    if setup is None or camera_index >= min(len(setup.cameras), len(paths)):
        return None
    camera = setup.cameras[camera_index]
    return camera if Path(bpy.path.abspath(paths[camera_index])).name == camera.video_file_name else None


def sync_videos_by_sound(video_paths: Sequence[Path]) -> list[AudioSync]:
    try:
        reference_samples, reference_rate = read_audio(video_paths[0])
        syncs = []
        for video_path in video_paths[1:]:
            samples, sample_rate = read_audio(video_path)
            syncs.append(find_start_delay(reference_samples, reference_rate, samples, sample_rate, _LARGEST_SYNC_DELAY_SECONDS))
        return syncs
    except (VideoAudioError, AudioSyncError) as error:
        raise CalibrationRequestError(str(error)) from error


def describe_syncs(syncs: Sequence[AudioSync]) -> tuple[str, bool]:
    parts = []
    all_clear = True
    for camera_index, sync in enumerate(syncs, start=2):
        is_clear = sync.peak_ratio >= _CLEAR_PEAK_RATIO
        all_clear &= is_clear
        parts.append(
            f"camera {camera_index} started {sync.other_started_later_seconds:+.3f} s after camera 1"
            + ("" if is_clear else " (unsure: clap once while both record, or set it by hand)")
        )
    return "Synced by sound: " + "; ".join(parts) + ".", all_clear


def calibration_request(properties: bpy.types.PropertyGroup) -> tuple[list[Path], CalibrationInputs]:
    if not properties.extra_videos:
        raise CalibrationRequestError("Add a second camera's video first.")
    if uses_body(properties):
        for camera_index in range(len(properties.extra_videos) + 1):
            if lens_is_missing_for_body(camera_settings(properties, camera_index)):
                raise CalibrationRequestError(
                    f"Camera {camera_index + 1}: calibrating from the body needs its lens. Type its focal length or pick a "
                    "saved lens."
                )
    paths = chosen_calibration_paths(properties)
    video_paths = [Path(bpy.path.abspath(path)) for path in paths]
    for camera_index, (path, video_path) in enumerate(zip(paths, video_paths), start=1):
        if not path or not video_path.is_file():
            raise CalibrationRequestError(f"Camera {camera_index}: video not found ({path or 'none chosen'}).")
    if properties.use_separate_calibration_videos:
        start_delays = [0.0] + [sync.other_started_later_seconds for sync in sync_videos_by_sound(video_paths)]
    else:
        start_delays = take_start_delays_seconds(properties)
    inputs = CalibrationInputs(
        videos=tuple(CalibrationVideo.of(video_path) for video_path in video_paths),
        start_delays_seconds=tuple(start_delays),
        saved_lenses=tuple(chosen_saved_lenses(properties)),
        typed_lenses=tuple(chosen_typed_lenses(properties)),
        square_size_metres=properties.board_square_millimetres / 1000.0,
        edge_below_squares_metres=properties.edge_below_squares_metres(),
        first_frame=properties.calibration_start_frame - 1,
        stop_frame=properties.calibration_end_frame,
        body=BodyCalibrationInputs(properties.cameras_apart_metres, properties.capture_settings())
        if uses_body(properties)
        else None,
    )
    return video_paths, inputs


def describe_lens(
    camera: CalibratedCamera, saved_lens: SavedLens | None, typed_lens: TypedLens | None, typed_lens_is_kept: bool
) -> str:
    if saved_lens is not None:
        return f"{saved_lens.name} (saved)"
    if typed_lens is not None and (typed_lens.keep_fixed or typed_lens_is_kept):
        return f"{typed_lens.focal_length_millimetres:g} mm (typed)"
    return f"{camera.focal_length_millimetres:.1f} mm ({camera.lens_error_pixels:.2f} px)"


def describe_setup(setup: CameraSetup) -> str:
    lens_errors = " / ".join(
        describe_lens(camera, saved_lens, typed_lens, setup.inputs.uses_body)
        for camera, saved_lens, typed_lens in zip(setup.cameras, setup.inputs.saved_lenses, setup.inputs.typed_lenses)
    )
    pair_errors = " / ".join(f"{error:.2f}" for error in setup.pair_error_pixels)
    if setup.floor_found and setup.floor_video_frames is not None:
        first, last = setup.floor_video_frames
        floor = (
            f"floor from {_FLOOR_DESCRIPTIONS[setup.floor_source]} (camera {setup.floor_camera_index + 1}, "
            f"frames {first + 1}-{last + 1})"
        )
    elif setup.floor_found:
        floor = f"floor from {_FLOOR_DESCRIPTIONS[setup.floor_source]}"
    else:
        floor = "no floor shot found, the floor will come from the feet"
    corrections = "".join(
        f" Camera {camera_index}'s sound offset corrected by {1000.0 * camera.picture_delay_correction_seconds:+.0f} ms."
        for camera_index, camera in enumerate(setup.cameras, start=1)
        if abs(camera.picture_delay_correction_seconds) >= 0.0005
    )
    return (
        f"{len(setup.cameras)} cameras calibrated: lenses {lens_errors}, cameras agree within {pair_errors} px; "
        f"{floor}.{corrections}"
    )


def apply_camera_setup(context: bpy.types.Context, setup: CameraSetup) -> str:
    scene = context.scene
    properties = scene.freeclipmocap
    calibration_state.set_setup(setup)
    place_phone_cameras(scene, setup, load_take_clips(properties))
    first_camera = setup.cameras[0]
    if (scene.render.resolution_x, scene.render.resolution_y) != (first_camera.image_width, first_camera.image_height):
        scene.render.resolution_x = first_camera.image_width
        scene.render.resolution_y = first_camera.image_height
        return f" Render size set to camera 1's ({first_camera.image_width}×{first_camera.image_height})."
    return ""


def reuse_saved_calibration(context: bpy.types.Context, video_paths: Sequence[Path], inputs: CalibrationInputs) -> str | None:
    saved_setup = find_current_setup(calibration_path_for(video_paths[0]), inputs)
    if saved_setup is None:
        return None
    message = f"Reused the saved calibration. {describe_setup(saved_setup)}{apply_camera_setup(context, saved_setup)}"
    calibration_state.set_status_message(message)
    return message


class AddCameraVideoOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.add_camera_video"
    bl_label = "Add Camera"
    bl_description = "Add another camera's video of the same take"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        context.scene.freeclipmocap.extra_videos.add()
        return {"FINISHED"}


class RemoveCameraVideoOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.remove_camera_video"
    bl_label = "Remove Camera"
    bl_description = "Remove this camera's video from the list (the file is not touched)"
    bl_options = {"REGISTER", "UNDO"}

    index: bpy.props.IntProperty(options={"HIDDEN"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        extra_videos = context.scene.freeclipmocap.extra_videos
        if 0 <= self.index < len(extra_videos):
            extra_videos.remove(self.index)
        return {"FINISHED"}


class SyncBySoundOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.sync_by_sound"
    bl_label = "Sync by Sound"
    bl_description = "Find how much later each camera started recording by matching the videos' sound (a clap helps)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        properties = context.scene.freeclipmocap
        return bool(properties.video_path) and len(properties.extra_videos) > 0

    def execute(self, context: bpy.types.Context) -> set[str]:
        properties = context.scene.freeclipmocap
        video_paths = take_video_paths(properties)
        for camera_index, video_path in enumerate(video_paths, start=1):
            if not video_path.is_file():
                self.report({"ERROR"}, f"Camera {camera_index}: video not found ({video_path}).")
                return {"CANCELLED"}
        try:
            syncs = sync_videos_by_sound(video_paths)
        except CalibrationRequestError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        for camera, sync in zip(properties.extra_videos, syncs):
            camera.started_later_seconds = sync.other_started_later_seconds
        message, all_clear = describe_syncs(syncs)
        self.report({"INFO"} if all_clear else {"WARNING"}, message)
        return {"FINISHED"}


class SaveBoardImageOperator(bpy.types.Operator, ExportHelper):
    bl_idname = "freeclipmocap.save_board_image"
    bl_label = "Save Board Image"
    bl_description = "Save the calibration board as an A4 page to print at 100% scale or show full-screen on a tablet"

    filename_ext = ".png"
    filter_glob: bpy.props.StringProperty(default="*.png", options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        self.filepath = "FreeClipMocap board.png"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context: bpy.types.Context) -> set[str]:
        from ..capture.board_detection import BoardDetectionError, save_board_image

        path = Path(self.filepath)
        try:
            save_board_image(path)
        except BoardDetectionError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Saved {path.name}. Print it at 100% scale or show it full-screen on a tablet, then measure one square "
            "and enter its size.",
        )
        return {"FINISHED"}


class SaveLensOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.save_lens"
    bl_label = "Save Lens"
    bl_description = (
        "Save this phone's lens from the last calibration in your FreeClipMocap user folder and use it for this camera: "
        "later calibrations keep it fixed and need no close-ups. It holds for the same phone, lens, resolution, "
        "orientation and zoom"
    )

    camera_index: bpy.props.IntProperty(options={"HIDDEN"})
    lens_name: bpy.props.StringProperty(name="Name")

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        camera = calibrated_camera(context, self.camera_index)
        if camera is None:
            self.report({"ERROR"}, _CALIBRATE_FIRST)
            return {"CANCELLED"}
        chosen_name = camera_settings(context.scene.freeclipmocap, self.camera_index).calibration_lens_name
        self.lens_name = chosen_name or f"Phone {self.camera_index + 1} {camera.image_width}x{camera.image_height}"
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context: bpy.types.Context) -> None:
        self.layout.prop(self, "lens_name")
        if not is_valid_lens_name(self.lens_name):
            self.layout.label(text="Use letters, digits, spaces, '-' and '_' only.", icon="ERROR")
        elif lens_path(user_lens_folder(), self.lens_name).exists():
            self.layout.label(text="A lens with this name exists and will be overwritten.", icon="INFO")

    def execute(self, context: bpy.types.Context) -> set[str]:
        camera = calibrated_camera(context, self.camera_index)
        if camera is None:
            self.report({"ERROR"}, _CALIBRATE_FIRST)
            return {"CANCELLED"}
        saved_lens = SavedLens.from_arrays(
            self.lens_name, camera.image_width, camera.image_height, camera.camera_matrix, camera.distortion, camera.lens_error_pixels
        )
        try:
            path = saved_lens.save(user_lens_folder())
        except (OSError, SavedLensError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        settings = camera_settings(context.scene.freeclipmocap, self.camera_index)
        settings.calibration_lens_is_typed = False
        settings.calibration_lens_name = saved_lens.name
        self.report({"INFO"}, f"Saved lens {saved_lens.name} for camera {self.camera_index + 1}: {path}")
        return {"FINISHED"}


class DeleteLensOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.delete_lens"
    bl_label = "Delete Lens"
    bl_description = "Delete this camera's saved lens from your FreeClipMocap user folder; cameras using it go back to Measure"

    camera_index: bpy.props.IntProperty(options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        lens_name = camera_settings(context.scene.freeclipmocap, self.camera_index).calibration_lens_name
        return context.window_manager.invoke_confirm(
            self, event, message=f"Delete the saved lens {lens_name}? Its file is removed for good.", confirm_text="Delete"
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        properties = context.scene.freeclipmocap
        lens_name = camera_settings(properties, self.camera_index).calibration_lens_name
        if not lens_name:
            return {"CANCELLED"}
        try:
            delete_lens(user_lens_folder(), lens_name)
        except (OSError, SavedLensError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        for camera_index in range(len(properties.extra_videos) + 1):
            settings = camera_settings(properties, camera_index)
            if settings.calibration_lens_name == lens_name:
                settings.calibration_lens_name = ""
        self.report({"INFO"}, f"Deleted the saved lens {lens_name}.")
        return {"FINISHED"}


class CalibrateOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.calibrate"
    bl_label = "Calibrate"
    bl_description = (
        "Find where the cameras stand (and each lens) from the board or the actor's body in the chosen frames, save it "
        "next to the first camera's video, and add a camera per phone with its video behind it. From the body, each "
        "video is captured first if it is not yet. A saved calibration from the same videos and settings is reused"
    )

    _timer: bpy.types.Timer | None = None

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return (
            not calibration_state.is_running
            and not capture_state.is_running
            and len(context.scene.freeclipmocap.extra_videos) > 0
        )

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        properties = context.scene.freeclipmocap
        try:
            video_paths, inputs = calibration_request(properties)
        except CalibrationRequestError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        reuse_message = reuse_saved_calibration(context, video_paths, inputs)
        if reuse_message is not None:
            self.report({"INFO"}, reuse_message)
            return {"FINISHED"}

        job = CalibrationJob(video_paths, inputs, calibration_path_for(video_paths[0]), MODELS_FOLDER)
        calibration_state.set_job(job)
        calibration_state.set_status_message("")
        job.start()
        self._timer = context.window_manager.event_timer_add(_TIMER_INTERVAL_SECONDS, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event) -> set[str]:
        job = calibration_state.job
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
        if calibration_state.job is not None:
            calibration_state.job.cancel()
        self._finish(context)

    def _finish(self, context: bpy.types.Context) -> None:
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        redraw_sidebars(context)

    def _report_result(self, context: bpy.types.Context, job: CalibrationJob) -> None:
        if job.setup is not None:
            message = f"{describe_setup(job.setup)}{apply_camera_setup(context, job.setup)}"
            self.report({"INFO"}, f"{message} Saved: {job.setup_path}")
        elif job.was_cancelled:
            message = "Calibration cancelled."
            self.report({"WARNING"}, message)
        else:
            message = f"Calibration failed: {job.error_message}"
            self.report({"ERROR"}, message)
        calibration_state.set_status_message(message)


class CancelCalibrationOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.cancel_calibration"
    bl_label = "Cancel"
    bl_description = "Stop the running calibration; nothing is saved"

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return calibration_state.is_running

    def execute(self, context: bpy.types.Context) -> set[str]:
        if calibration_state.job is not None:
            calibration_state.job.cancel()
        return {"FINISHED"}


def cancel_running_calibration() -> None:
    if calibration_state.job is not None:
        calibration_state.job.cancel()
        calibration_state.job.wait()
    calibration_state.set_job(None)


_classes = (
    AddCameraVideoOperator,
    RemoveCameraVideoOperator,
    SyncBySoundOperator,
    SaveBoardImageOperator,
    SaveLensOperator,
    DeleteLensOperator,
    CalibrateOperator,
    CancelCalibrationOperator,
)


def register() -> None:
    for operator_class in _classes:
        bpy.utils.register_class(operator_class)


def unregister() -> None:
    cancel_running_calibration()
    for operator_class in reversed(_classes):
        bpy.utils.unregister_class(operator_class)
