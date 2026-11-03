from collections.abc import Sequence

import bpy
import numpy as np
from mathutils import Matrix

from ..capture.camera_setup import CalibratedCamera, CameraSetup
from ..constants.camera_lens import FULL_FRAME_WIDTH_MILLIMETRES

_CAMERA_NAME_PREFIX = "FreeClipMocap Phone"


def phone_camera_name(camera_index: int) -> str:
    return f"{_CAMERA_NAME_PREFIX} {camera_index + 1}"


def place_phone_cameras(
    scene: bpy.types.Scene,
    setup: CameraSetup,
    take_clips: Sequence[bpy.types.MovieClip | None],
) -> list[bpy.types.Object]:
    camera_objects = []
    for camera_index, camera in enumerate(setup.cameras):
        camera_object = _phone_camera(scene, phone_camera_name(camera_index))
        _place(camera_object, camera)
        if camera_index < len(take_clips) and take_clips[camera_index] is not None:
            _show_video_behind(camera_object, take_clips[camera_index])
        camera_objects.append(camera_object)
    return camera_objects


def _phone_camera(scene: bpy.types.Scene, name: str) -> bpy.types.Object:
    camera_object = bpy.data.objects.get(name)
    if camera_object is None or camera_object.type != "CAMERA":
        camera_object = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    if camera_object.name not in scene.objects:
        scene.collection.objects.link(camera_object)
    return camera_object


def _place(camera_object: bpy.types.Object, camera: CalibratedCamera) -> None:
    # OpenCV's camera looks along +z with y down; Blender's looks along -z with y up.
    right, down, forward = camera.rotation
    position = camera.position
    camera_object.matrix_world = Matrix((
        (right[0], -down[0], -forward[0], position[0]),
        (right[1], -down[1], -forward[1], position[1]),
        (right[2], -down[2], -forward[2], position[2]),
        (0.0, 0.0, 0.0, 1.0),
    ))
    larger_side = max(camera.image_width, camera.image_height)
    camera_data = camera_object.data
    camera_data.type = "PERSP"
    camera_data.sensor_fit = "AUTO"
    camera_data.sensor_width = FULL_FRAME_WIDTH_MILLIMETRES
    camera_data.lens = camera.focal_length_millimetres
    camera_data.shift_x = (camera.image_width / 2.0 - float(camera.camera_matrix[0, 2])) / larger_side
    camera_data.shift_y = (float(camera.camera_matrix[1, 2]) - camera.image_height / 2.0) / larger_side
    camera_data.clip_end = max(camera_data.clip_end, 3.0 * float(np.linalg.norm(position)))


def _show_video_behind(camera_object: bpy.types.Object, clip: bpy.types.MovieClip) -> None:
    camera_data = camera_object.data
    camera_data.show_background_images = True
    background = camera_data.background_images[0] if camera_data.background_images else camera_data.background_images.new()
    background.source = "MOVIE_CLIP"
    background.use_camera_clip = False
    background.clip = clip
    background.frame_method = "STRETCH"
    background.display_depth = "BACK"
    background.alpha = 1.0
