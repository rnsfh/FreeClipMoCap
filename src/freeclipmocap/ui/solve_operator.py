from pathlib import Path

import bpy

from ..animation.action_writer import write_take
from ..animation.frame_rates import nearest_standard_frame_rate
from ..animation.ik_controls import has_ik_controls, key_ik_targets_from_fk, muted_ik_controls
from ..capture.landmark_cache import CacheFormatError, LandmarkCache, find_current_cache
from ..constants.camera_lens import FULL_FRAME_WIDTH_MILLIMETRES
from ..rig.armature_bone_map import apply_rig_settings, read_bone_map
from ..rig.armature_rest import rig_rest_from_armature
from ..solver.leg_depth import with_legs_from_the_video
from ..solver.mediapipe_table import MEDIAPIPE_TABLE
from ..solver.pose_solver import SolveError, SolveSettings, solve
from ..solver.triangulation import TriangulationError, TriangulationReport, triangulated_cache
from .calibration_operators import CalibrationRequestError, current_camera_setup
from .capture_operators import capture_state
from .video_operators import scene_frame_of, take_start_delays_seconds, take_video_paths


def uses_all_cameras(properties: bpy.types.PropertyGroup) -> bool:
    return bool(properties.extra_videos) and properties.use_all_cameras


def describe_triangulation(report: TriangulationReport, camera_count: int) -> str:
    left, right = report.left_hand, report.right_hand
    swapped = sum(report.swapped_frame_counts)
    return (
        f" Triangulated from {camera_count} cameras: body in {report.body_share:.0%} of frames, the cameras agreeing "
        f"within {report.median_error_pixels:.1f} px (median); hands from several cameras {left.triangulated_share:.0%} / "
        f"{right.triangulated_share:.0%}, from one {left.one_camera_share:.0%} / {right.one_camera_share:.0%}"
        + (f"; left and right undone in {swapped} frame(s)." if swapped else ".")
    )


def leave_edit_mode(context: bpy.types.Context, armature_object: bpy.types.Object) -> bool:
    if armature_object.mode != "EDIT":
        return True
    try:
        with context.temp_override(active_object=armature_object, object=armature_object):
            bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        return False
    return armature_object.mode != "EDIT"


class SolveOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.solve"
    bl_label = "Solve"
    bl_description = (
        "Turn the captured landmarks of the chosen video frames into a new Action (take) on the target armature; "
        "earlier takes are kept. The take sits under the same video frames, the scene is set to the video's frame "
        "rate and range, and the armature is selected so the Action Editor shows the take"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        properties = context.scene.freeclipmocap
        return (
            not capture_state.is_running
            and bool(properties.video_path)
            and properties.target_armature is not None
        )

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene = context.scene
        properties = scene.freeclipmocap
        video_paths = take_video_paths(properties) if uses_all_cameras(properties) else take_video_paths(properties)[:1]
        video_path = video_paths[0]
        settings = properties.capture_settings()
        cache_paths = [find_current_cache(path, settings) if path.is_file() else None for path in video_paths]
        missing = [str(camera_number) for camera_number, path in enumerate(cache_paths, start=1) if path is None]
        if len(video_paths) == 1 and missing:
            self.report({"ERROR"}, "No capture of this video with these settings yet. Press Capture first.")
            return {"CANCELLED"}
        if missing:
            self.report(
                {"ERROR"},
                f"No capture of camera {', '.join(missing)} with these settings yet. Press Capture; it captures every camera.",
            )
            return {"CANCELLED"}
        armature_object = properties.target_armature
        edit_mode_note = f" Left Edit Mode on {armature_object.name} so the take plays." if armature_object.mode == "EDIT" else ""
        if not leave_edit_mode(context, armature_object):
            self.report({"ERROR"}, f"Leave Edit Mode on {armature_object.name} first (Tab); its pose does not play in Edit Mode.")
            return {"CANCELLED"}
        camera_note = ""
        try:
            caches = [LandmarkCache.load(cache_path) for cache_path in cache_paths]
            cache = caches[0]
            if len(caches) > 1:
                setup = current_camera_setup(properties)
                if setup is None:
                    self.report(
                        {"ERROR"},
                        "The cameras are not calibrated for these videos, Sync and lenses. Press Calibrate first, or turn "
                        "All Cameras off to solve from camera 1.",
                    )
                    return {"CANCELLED"}
                cache, report = triangulated_cache(caches, setup.cameras, take_start_delays_seconds(properties))
                camera_note = describe_triangulation(report, len(caches))
            else:
                larger_side = max(cache.video.width, cache.video.height)
                focal_length_pixels = properties.compare_lens_millimetres * larger_side / FULL_FRAME_WIDTH_MILLIMETRES
                cache = with_legs_from_the_video(cache, focal_length_pixels)
            first_frame = properties.video_start_frame
            last_frame = properties.video_end_frame or cache.frame_count
            if last_frame > cache.frame_count:
                last_frame = cache.frame_count
                properties.video_end_frame = last_frame
            if first_frame > last_frame:
                self.report({"ERROR"}, f"Start frame {first_frame} is after the end frame {last_frame}.")
                return {"CANCELLED"}
            take = solve(
                cache.sliced(first_frame - 1, last_frame),
                rig_rest_from_armature(armature_object),
                read_bone_map(armature_object),
                MEDIAPIPE_TABLE,
                SolveSettings(
                    smoothing=properties.smoothing, root_motion=properties.root_motion, level_by_body=len(caches) == 1
                ),
            )
        except (CacheFormatError, SolveError, CalibrationRequestError, TriangulationError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        frame_rate_note = ""
        frames_per_second, frame_rate_base = nearest_standard_frame_rate(cache.video.frame_rate)
        if (scene.render.fps, round(scene.render.fps_base, 4)) != (frames_per_second, round(frame_rate_base, 4)):
            scene.render.fps = frames_per_second
            scene.render.fps_base = frame_rate_base
            frame_rate_note = " Scene frame rate set to the video's."
        frame_rate = scene.render.fps / scene.render.fps_base
        take = take.resampled(frame_rate)
        key_tolerance = properties.key_tolerance_degrees if properties.reduce_keys else None
        take_start_frame = scene_frame_of(first_frame)
        for change in apply_rig_settings(armature_object):
            self.report({"INFO"}, f"Rig setting for mocap: {change}")
        with muted_ik_controls(armature_object):
            written = write_take(armature_object, take, video_path.stem, take_start_frame, key_tolerance)
        action = written.action
        if written.unmatched_bones:
            self.report(
                {"WARNING"},
                "The rig's constraints keep these bones from following the take exactly: "
                + ", ".join(written.unmatched_bones)
                + ".",
            )
        scene.frame_start = take_start_frame
        scene.frame_end = take_start_frame + take.frame_count - 1
        ik_note = ""
        if has_ik_controls(armature_object):
            key_ik_targets_from_fk(scene, armature_object)
            ik_note = " IK targets keyed to follow the new take."
        scene.frame_set(scene.frame_start)
        if armature_object.name in context.view_layer.objects:
            armature_object.select_set(True)
            context.view_layer.objects.active = armature_object
        for warning in take.warnings:
            self.report({"WARNING"}, warning)
        self.report(
            {"INFO"},
            f"Solved take {action.name} from video frames {first_frame}-{last_frame}: "
            f"{len(take.rotations_by_bone)} bones, {take.frame_count} frames at "
            f"{frame_rate:.3g} fps"
            + (f"; camera tilt corrected by {take.camera_tilt_degrees:.0f}°." if not camera_note else ".")
            + f"{frame_rate_note}{edit_mode_note}{camera_note}{ik_note}",
        )
        return {"FINISHED"}


register, unregister = bpy.utils.register_classes_factory((SolveOperator,))
