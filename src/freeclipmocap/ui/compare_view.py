from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

from ..capture.landmark_cache import CacheFormatError, LandmarkCache, find_current_cache
from ..rig.armature_bone_map import read_bone_map
from ..solver.camera_fit import CameraFitError, FittedCamera, fit_camera
from ..solver.mediapipe_table import MEDIAPIPE_TABLE
from ..solver.tracking_table import LandmarkPoint, PointSet
from .capture_operators import capture_state
from .solve_operator import leave_edit_mode
from .video_operators import load_video_clip, video_frame_at

_CAMERA_NAME = "FreeClipMocap Camera"
_SAMPLED_FRAMES = 60
_SENSOR_WIDTH_MILLIMETRES = 36.0
_WORLD_UP = np.array([0.0, 0.0, 1.0])


def _tracked_pixels(cache: LandmarkCache, point: LandmarkPoint, frame_index: int, minimum_visibility: float) -> np.ndarray | None:
    indices = list(point.indices)
    if point.point_set is PointSet.POSE:
        if np.min(cache.pose.visibility[frame_index, indices]) < minimum_visibility:
            return None
        landmarks = cache.pose.image_landmarks
    elif point.point_set is PointSet.LEFT_HAND:
        landmarks = cache.left_hand.image_landmarks
    else:
        landmarks = cache.right_hand.image_landmarks
    position = np.mean(landmarks[frame_index, indices, :2], axis=0)
    if np.any(np.isnan(position)):
        return None
    return position * (cache.video.width, cache.video.height)


def _paired_points(
    scene: bpy.types.Scene, armature_object: bpy.types.Object, cache: LandmarkCache
) -> tuple[np.ndarray, np.ndarray, int]:
    bone_map = read_bone_map(armature_object)
    joints = [
        (bone_name, joint.point)
        for joint in MEDIAPIPE_TABLE.joint_points
        if (bone_name := bone_map.bone_for(joint.role)) is not None and bone_name in armature_object.pose.bones
    ]
    first_frame, last_frame = (int(round(frame)) for frame in armature_object.animation_data.action.frame_range)
    scene_frames = np.unique(np.linspace(first_frame, last_frame, _SAMPLED_FRAMES).round().astype(int))
    world_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_frames = 0
    frame_before = scene.frame_current
    for scene_frame in scene_frames:
        frame_index = video_frame_at(int(scene_frame)) - 1
        if not 0 <= frame_index < cache.frame_count or not cache.pose.found[frame_index]:
            continue
        scene.frame_set(int(scene_frame))
        pairs_before = len(world_points)
        for bone_name, point in joints:
            pixels = _tracked_pixels(cache, point, frame_index, MEDIAPIPE_TABLE.minimum_visibility)
            if pixels is None:
                continue
            world_points.append(np.array(armature_object.matrix_world @ armature_object.pose.bones[bone_name].head))
            image_points.append(pixels)
        used_frames += len(world_points) > pairs_before
    scene.frame_set(frame_before)
    return np.array(world_points).reshape(-1, 3), np.array(image_points).reshape(-1, 2), used_frames


def _compare_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    properties = scene.freeclipmocap
    camera_object = properties.compare_camera
    if camera_object is None or camera_object.type != "CAMERA":
        camera_object = bpy.data.objects.new(_CAMERA_NAME, bpy.data.cameras.new(_CAMERA_NAME))
        properties.compare_camera = camera_object
    if camera_object.name not in scene.objects:
        scene.collection.objects.link(camera_object)
    return camera_object


def _place_camera(
    camera_object: bpy.types.Object, fitted: FittedCamera, lens_millimetres: float, clip: bpy.types.MovieClip
) -> None:
    rotation = fitted.rotation
    position = fitted.position
    camera_object.matrix_world = Matrix((
        (*rotation[0], position[0]),
        (*rotation[1], position[1]),
        (*rotation[2], position[2]),
        (0.0, 0.0, 0.0, 1.0),
    ))
    camera_data = camera_object.data
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "AUTO"
    camera_data.sensor_width = _SENSOR_WIDTH_MILLIMETRES
    camera_data.lens = lens_millimetres
    camera_data.shift_x = 0.0
    camera_data.shift_y = 0.0
    camera_data.clip_end = max(camera_data.clip_end, 3.0 * float(np.linalg.norm(position)))
    camera_data.show_background_images = True
    background = camera_data.background_images[0] if camera_data.background_images else camera_data.background_images.new()
    background.source = "MOVIE_CLIP"
    background.use_camera_clip = False
    background.clip = clip
    background.frame_method = "STRETCH"
    background.display_depth = "BACK"
    background.alpha = 1.0


def _look_through(context: bpy.types.Context, camera_object: bpy.types.Object) -> None:
    screen = context.screen
    if screen is None:
        return
    viewports = [area for area in screen.areas if area.type == "VIEW_3D"]
    if not viewports:
        return
    area = max(viewports, key=lambda area: area.width * area.height)
    space = area.spaces.active
    space.use_local_camera = True
    space.camera = camera_object
    space.region_3d.view_perspective = "CAMERA"
    region = next(region for region in area.regions if region.type == "WINDOW")
    try:
        with context.temp_override(area=area, region=region):
            bpy.ops.view3d.view_center_camera()
    except RuntimeError:
        pass


class CompareViewOperator(bpy.types.Operator):
    bl_idname = "freeclipmocap.compare_view"
    bl_label = "Compare View"
    bl_description = (
        "Look at the take through a camera fitted to the video, with the video behind the rig. Adds or updates the "
        f"camera '{_CAMERA_NAME}', sets the render size to the video's and shows it in the largest 3D viewport"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        properties = context.scene.freeclipmocap
        armature_object = properties.target_armature
        return (
            not capture_state.is_running
            and bool(properties.video_path)
            and armature_object is not None
            and armature_object.animation_data is not None
            and armature_object.animation_data.action is not None
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
        if not leave_edit_mode(context, armature_object):
            self.report({"ERROR"}, f"Leave Edit Mode on {armature_object.name} first (Tab); its pose does not play in Edit Mode.")
            return {"CANCELLED"}
        try:
            cache = LandmarkCache.load(cache_path)
            world_points, image_points, used_frames = _paired_points(scene, armature_object, cache)
            focal_length_pixels = (
                properties.compare_lens_millimetres * max(cache.video.width, cache.video.height) / _SENSOR_WIDTH_MILLIMETRES
            )
            fitted = fit_camera(world_points, image_points, cache.video.width, cache.video.height, focal_length_pixels, _WORLD_UP)
        except (CacheFormatError, CameraFitError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        camera_object = _compare_camera(scene)
        _place_camera(camera_object, fitted, properties.compare_lens_millimetres, load_video_clip(video_path))
        resolution_note = ""
        if (scene.render.resolution_x, scene.render.resolution_y) != (cache.video.width, cache.video.height):
            scene.render.resolution_x = cache.video.width
            scene.render.resolution_y = cache.video.height
            resolution_note = f" Render size set to the video's ({cache.video.width}×{cache.video.height})."
        _look_through(context, camera_object)
        root_motion_note = (
            ""
            if properties.root_motion
            else " Root Motion is off, so the rig stays in place while the actor travels in the video."
        )
        self.report(
            {"INFO"},
            f"Camera fitted on {used_frames} frames: the rig's joints land a median {fitted.median_error_pixels:.0f} px "
            f"from the tracked points ({100.0 * fitted.median_error_pixels / cache.video.height:.1f}% of the video "
            f"height).{resolution_note}{root_motion_note}",
        )
        return {"FINISHED"}


register, unregister = bpy.utils.register_classes_factory((CompareViewOperator,))
