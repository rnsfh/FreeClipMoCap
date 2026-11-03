from pathlib import Path

import bpy

from ..animation.action_writer import write_take
from ..animation.frame_rates import nearest_standard_frame_rate
from ..capture.landmark_cache import CacheFormatError, LandmarkCache, find_current_cache
from ..rig.armature_bone_map import apply_rig_settings, read_bone_map
from ..rig.armature_rest import rig_rest_from_armature
from ..solver.mediapipe_table import MEDIAPIPE_TABLE
from ..solver.pose_solver import SolveError, SolveSettings, solve
from .capture_operators import capture_state
from .video_operators import scene_frame_of


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
        video_path = Path(bpy.path.abspath(properties.video_path))
        cache_path = find_current_cache(video_path, properties.capture_settings()) if video_path.is_file() else None
        if cache_path is None:
            self.report({"ERROR"}, "No capture of this video with these settings yet. Press Capture first.")
            return {"CANCELLED"}
        armature_object = properties.target_armature
        try:
            cache = LandmarkCache.load(cache_path)
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
                SolveSettings(smoothing=properties.smoothing, root_motion=properties.root_motion),
            )
        except (CacheFormatError, SolveError) as error:
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
            f"{frame_rate:.3g} fps; camera tilt corrected by {take.camera_tilt_degrees:.0f}°.{frame_rate_note}",
        )
        return {"FINISHED"}


register, unregister = bpy.utils.register_classes_factory((SolveOperator,))
