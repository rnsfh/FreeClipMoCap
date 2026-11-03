import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from bl_ext.user_default.freeclipmocap.capture.landmark_cache import LandmarkCache, VideoSource, cache_path_for
from bl_ext.user_default.freeclipmocap.solver.camera_fit import project
from bl_ext.user_default.freeclipmocap.solver.mediapipe_table import MEDIAPIPE_TABLE
from bl_ext.user_default.freeclipmocap.ui.video_operators import scene_frame_of

from ..animation.test_action_writer import create_armature, remove_armature
from ..solver.synthetic_actor import make_cache, make_rig, rest_landmarks, rotation_about

_WIDTH = 640
_HEIGHT = 480
_CAMERA_POSITION = np.array([2.0, -3.0, 1.4])
_CAMERA_TARGET = np.array([0.0, 0.0, 1.0])
_FOCAL_LENGTH_PIXELS = 700.0


def camera_rotation() -> np.ndarray:
    back = (_CAMERA_POSITION - _CAMERA_TARGET) / np.linalg.norm(_CAMERA_POSITION - _CAMERA_TARGET)
    right = np.cross((0.0, 0.0, 1.0), back)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(back, right), back))


class CompareViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_folder = tempfile.TemporaryDirectory()
        rig, _bone_map = make_rig()
        bones = [
            (name, None if parent < 0 else rig.bone_names[parent], tuple(head), tuple(tail))
            for name, parent, head, tail in zip(rig.bone_names, rig.parent_indices, rig.heads, rig.tails)
        ]
        self.armature_object = create_armature("SmartRigArmature", bones)
        rest = rest_landmarks(rig)
        bent = {name: points.copy() for name, points in rest.items()}
        elbow = rest["pose"][13]
        bend = rotation_about((0.0, 0.0, 1.0), -90.0)
        bent["pose"][[15, 17, 19, 21]] = (rest["pose"][[15, 17, 19, 21]] - elbow) @ bend.T + elbow
        bent["Left"] = (rest["Left"] - elbow) @ bend.T + elbow
        self.video_path = Path(self.temporary_folder.name) / "clip.mp4"
        self.video_path.write_bytes(b"stand-in video")
        status = self.video_path.stat()
        synthetic = make_cache([rest] * 15 + [bent] * 16)
        self.cache = LandmarkCache(
            tracking_model=synthetic.tracking_model,
            settings=synthetic.settings,
            video=VideoSource(status.st_size, status.st_mtime_ns, 30.0, _WIDTH, _HEIGHT),
            frame_indices=synthetic.frame_indices,
            timestamps_milliseconds=synthetic.timestamps_milliseconds,
            pose=synthetic.pose,
            left_hand=synthetic.left_hand,
            right_hand=synthetic.right_hand,
        )
        self.cache.save(cache_path_for(self.video_path, self.cache.settings))
        properties = bpy.context.scene.freeclipmocap
        properties.video_path = str(self.video_path)
        properties.pose_model_variant = self.cache.settings.pose_model_variant.value
        properties.smoothing = 0.0
        properties.target_armature = self.armature_object
        properties.compare_lens_millimetres = _FOCAL_LENGTH_PIXELS * 36.0 / _WIDTH
        scene = bpy.context.scene
        self.resolution_before = (scene.render.resolution_x, scene.render.resolution_y)
        self.actions_before = set(bpy.data.actions)
        self.objects_before = set(bpy.data.objects)

    def tearDown(self) -> None:
        scene = bpy.context.scene
        properties = scene.freeclipmocap
        properties.target_armature = None
        properties.compare_camera = None
        remove_armature(self.armature_object)
        for camera_object in set(bpy.data.objects) - self.objects_before:
            camera_data = camera_object.data
            bpy.data.objects.remove(camera_object)
            bpy.data.cameras.remove(camera_data)
        for action in set(bpy.data.actions) - self.actions_before:
            bpy.data.actions.remove(action)
        for clip in list(bpy.data.movieclips):
            bpy.data.movieclips.remove(clip)
        scene.render.resolution_x, scene.render.resolution_y = self.resolution_before
        self.temporary_folder.cleanup()

    def _film_the_take_with_a_known_camera(self) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
        scene = bpy.context.scene
        rotation = camera_rotation()
        image_landmarks = self.cache.pose.image_landmarks
        filmed: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for frame_index in range(self.cache.frame_count):
            scene.frame_set(scene_frame_of(frame_index + 1))
            filmed[frame_index] = []
            for joint in MEDIAPIPE_TABLE.joint_points:
                world_point = np.array(
                    self.armature_object.matrix_world @ self.armature_object.pose.bones[f"mixamorig:{joint.role.value}"].head
                )
                pixels = project(world_point[None], rotation, _CAMERA_POSITION, _FOCAL_LENGTH_PIXELS, _WIDTH, _HEIGHT)[0]
                image_landmarks[frame_index, joint.point.indices[0], :2] = pixels / (_WIDTH, _HEIGHT)
                filmed[frame_index].append((world_point, pixels))
        self.cache.save(cache_path_for(self.video_path, self.cache.settings))
        return filmed

    def test_the_fitted_camera_shows_the_rig_where_the_video_shows_the_actor(self) -> None:
        self.assertEqual(bpy.ops.freeclipmocap.solve(), {"FINISHED"})
        filmed = self._film_the_take_with_a_known_camera()

        self.assertEqual(bpy.ops.freeclipmocap.compare_view(), {"FINISHED"})

        scene = bpy.context.scene
        camera_object = scene.freeclipmocap.compare_camera
        self.assertIsNotNone(camera_object)
        self.assertEqual((scene.render.resolution_x, scene.render.resolution_y), (_WIDTH, _HEIGHT))
        self.assertLess(np.linalg.norm(np.array(camera_object.matrix_world.translation) - _CAMERA_POSITION), 0.02)
        self.assertAlmostEqual(camera_object.data.lens, 39.375, places=3)
        background = camera_object.data.background_images[0]
        self.assertEqual(Path(bpy.path.abspath(background.clip.filepath)).name, "clip.mp4")
        for frame_index in (0, 30):
            for world_point, pixels in filmed[frame_index]:
                seen = world_to_camera_view(scene, camera_object, Vector(world_point))
                np.testing.assert_allclose((seen.x * _WIDTH, (1.0 - seen.y) * _HEIGHT), pixels, atol=1.0)

    def test_pressing_again_updates_the_same_camera(self) -> None:
        self.assertEqual(bpy.ops.freeclipmocap.solve(), {"FINISHED"})
        self._film_the_take_with_a_known_camera()
        self.assertEqual(bpy.ops.freeclipmocap.compare_view(), {"FINISHED"})
        first_camera = bpy.context.scene.freeclipmocap.compare_camera
        self.assertEqual(bpy.ops.freeclipmocap.compare_view(), {"FINISHED"})
        self.assertIs(bpy.context.scene.freeclipmocap.compare_camera, first_camera)
        self.assertEqual(len([item for item in bpy.data.objects if item.type == "CAMERA"]) - len(
            [item for item in self.objects_before if item.type == "CAMERA"]
        ), 1)


if __name__ == "__main__":
    unittest.main()
