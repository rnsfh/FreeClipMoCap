import struct
import tempfile
import unittest
from pathlib import Path

import bpy
import cv2
import numpy as np
from io_scene_fbx import parse_fbx

from bl_ext.user_default.freeclipmocap.animation.action_writer import write_take
from bl_ext.user_default.freeclipmocap.animation.export import export_bvh, export_fbx
from bl_ext.user_default.freeclipmocap.animation.frame_rates import nearest_standard_frame_rate
from bl_ext.user_default.freeclipmocap.animation.key_reduction import select_key_frames, worst_frames_of_each_run
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolvedTake
from bl_ext.user_default.freeclipmocap.ui.video_operators import load_video_clip

from ..capture.video_fixtures import write_moving_shapes_video
from .test_action_writer import create_armature, remove_armature


def fbx_unit_scale_factor(path: Path) -> float:
    data = path.read_bytes()
    name = b"UnitScaleFactor"
    cursor = data.find(b"S" + struct.pack("<I", len(name)) + name) + 5 + len(name)
    while data[cursor:cursor + 1] == b"S":
        cursor += 5 + struct.unpack("<I", data[cursor + 1:cursor + 5])[0]
    return struct.unpack("<d", data[cursor + 1:cursor + 9])[0]


def child_element(element: parse_fbx.FBXElem, name: bytes) -> parse_fbx.FBXElem:
    return next(sub_element for sub_element in element.elems if sub_element.id == name)


def model_properties(model: parse_fbx.FBXElem) -> dict[str, list[float]]:
    return {
        entry.props[0].decode(): list(entry.props[4:])
        for entry in child_element(model, b"Properties70").elems
    }


def rotations_about(axis: tuple[float, float, float], angles_degrees: np.ndarray) -> np.ndarray:
    unit_axis = np.asarray(axis) / np.linalg.norm(axis)
    halves = np.radians(angles_degrees) / 2.0
    return np.concatenate((np.cos(halves)[:, None], np.sin(halves)[:, None] * unit_axis), axis=-1)


class FrameRateTests(unittest.TestCase):
    def test_phone_rates_snap_to_standard_rates(self) -> None:
        self.assertEqual(nearest_standard_frame_rate(30.08), (30, 1.0))
        self.assertEqual(nearest_standard_frame_rate(29.97), (30, 1.001))
        self.assertEqual(nearest_standard_frame_rate(59.6), (60, 1.001))
        self.assertEqual(nearest_standard_frame_rate(25.0), (25, 1.0))


class KeyReductionTests(unittest.TestCase):
    def test_still_rotation_needs_only_the_ends(self) -> None:
        quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (50, 1))
        np.testing.assert_array_equal(select_key_frames(quaternions, 0.5), [0, 49])

    def test_turn_and_back_keeps_the_turning_point(self) -> None:
        angles = np.radians(np.concatenate((np.linspace(0, 90, 26), np.linspace(90, 0, 26)[1:])))
        quaternions = np.stack((np.cos(angles / 2), np.zeros(51), np.zeros(51), np.sin(angles / 2)), axis=-1)
        self.assertIn(25, select_key_frames(quaternions, 0.5))

    def test_worst_frame_of_each_bad_run(self) -> None:
        errors = np.array([0.0, 0.2, 0.5, 0.3, 0.0, 0.0, 0.4, 0.0])
        np.testing.assert_array_equal(worst_frames_of_each_run(errors, 0.1), [2, 6])


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_folder.name)
        self.armature_object = create_armature(
            "ExportRig",
            [
                ("mixamorig:Hips", None, (0.0, 0.0, 1.0), (0.0, 0.0, 1.1)),
                ("mixamorig:Spine", "mixamorig:Hips", (0.0, 0.0, 1.1), (0.0, 0.0, 1.3)),
                ("mixamorig:LeftArm", "mixamorig:Spine", (0.05, 0.0, 1.3), (0.35, 0.05, 1.25)),
            ],
        )
        self.take = SolvedTake(
            np.arange(10) / 30.0,
            {
                "mixamorig:Spine": rotations_about((0.3, 0.5, 0.8), np.linspace(0.0, 120.0, 10)),
                "mixamorig:LeftArm": rotations_about((-0.6, 0.2, 0.4), np.linspace(0.0, -150.0, 10)),
            },
            (),
            0.0,
        )
        self.action = write_take(self.armature_object, self.take, "ExportTake", 1, None).action
        bpy.context.scene.render.fps = 30
        bpy.context.scene.render.fps_base = 1.0
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = 10
        self.objects_before = set(bpy.data.objects)
        self.actions_before = set(bpy.data.actions)
        self.armatures_before = set(bpy.data.armatures)

    def tearDown(self) -> None:
        for obj in set(bpy.data.objects) - self.objects_before:
            bpy.data.objects.remove(obj)
        for action in set(bpy.data.actions) - self.actions_before:
            bpy.data.actions.remove(action)
        for armature in set(bpy.data.armatures) - self.armatures_before:
            bpy.data.armatures.remove(armature)
        remove_armature(self.armature_object)
        bpy.data.actions.remove(self.action)
        self.temporary_folder.cleanup()

    def test_fbx_is_scaled_like_a_mixamo_clip(self) -> None:
        path = self.folder / "take.fbx"
        export_fbx(bpy.context, self.armature_object, path)
        self.assertAlmostEqual(fbx_unit_scale_factor(path), 1.0, places=6)
        root, _version = parse_fbx.parse(str(path))
        models = {
            model.props[1].split(b"\x00\x01")[0].decode(): model
            for model in child_element(root, b"Objects").elems
            if model.id == b"Model"
        }
        self.assertEqual(set(models), {"ExportRig", "mixamorig:Hips", "mixamorig:Spine", "mixamorig:LeftArm"})
        self.assertEqual(model_properties(models["ExportRig"]).get("Lcl Scaling", [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0])
        hips_translation = model_properties(models["mixamorig:Hips"])["Lcl Translation"]
        self.assertAlmostEqual(float(np.linalg.norm(hips_translation)), 1.0, places=4)

    def test_fbx_take_is_named_after_the_action_and_starts_at_zero(self) -> None:
        path = self.folder / "take.fbx"
        scene = bpy.context.scene
        scene_name = scene.name
        export_fbx(bpy.context, self.armature_object, path)
        root, _version = parse_fbx.parse(str(path))
        objects = child_element(root, b"Objects")
        stacks = [element.props[1].split(b"\x00\x01")[0].decode() for element in objects.elems if element.id == b"AnimationStack"]
        self.assertEqual(stacks, ["ExportTake"])
        first_key_times = {child_element(element, b"KeyTime").props[0][0] for element in objects.elems if element.id == b"AnimationCurve"}
        self.assertEqual(first_key_times, {0})
        self.assertEqual((scene.name, scene.frame_start, scene.frame_end), (scene_name, 1, 10))
        curve = self.action.fcurve_ensure_for_datablock(self.armature_object, 'pose.bones["mixamorig:Spine"].rotation_quaternion', index=0)
        self.assertEqual(curve.keyframe_points[0].co.x, 1.0)

    def test_bvh_is_y_up_and_imports_back_to_the_same_pose(self) -> None:
        path = self.folder / "take.bvh"
        export_bvh(bpy.context, self.armature_object, path)
        text = path.read_text()
        self.assertIn("ROOT mixamorig:Hips", text)
        self.assertIn("OFFSET 0.000000 1.000000 0.000000", text)
        self.assertEqual(text.count("CHANNELS 6"), 1)
        self.assertIn("CHANNELS 3 Zrotation Xrotation Yrotation", text)
        self.assertIn("Frames: 10\nFrame Time: 0.033333", text)
        objects_before_import = set(bpy.data.objects)
        bpy.ops.import_anim.bvh(filepath=str(path))
        imported = next(iter(set(bpy.data.objects) - objects_before_import))
        for frame in range(1, 11):
            bpy.context.scene.frame_set(frame)
            original_bones = self.armature_object.pose.bones
            imported_bones = imported.pose.bones
            points = [(original_bones[bone.name].head, imported_bones[bone.name].head) for bone in self.armature_object.data.bones]
            points.append((original_bones["mixamorig:LeftArm"].tail, imported_bones["mixamorig:LeftArm"].tail))
            for original_point, imported_point in points:
                difference = self.armature_object.matrix_world @ original_point - imported.matrix_world @ imported_point
                self.assertLess(difference.length, 1e-4, f"frame {frame}")


class VideoClipTests(unittest.TestCase):
    def test_video_frame_one_is_at_scene_frame_one(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            video_path = write_moving_shapes_video(Path(folder) / "clip.mp4", frame_count=12)
            clip = load_video_clip(video_path)
            try:
                self.assertEqual(clip.frame_start, 1)
                self.assertEqual(clip.frame_duration, 12)
                self.assertIs(load_video_clip(video_path), clip)
            finally:
                bpy.data.movieclips.remove(clip)
        self.assertTrue(cv2.__version__)

    def test_a_later_camera_s_clip_starts_by_its_sync_offset(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            video_path = write_moving_shapes_video(Path(folder) / "side.mp4", frame_count=12)
            clip = load_video_clip(video_path, started_later_seconds=0.37)
            try:
                self.assertEqual(clip.frame_start, 1 + round(0.37 * clip.fps))
                self.assertEqual(load_video_clip(video_path).frame_start, 1)
            finally:
                bpy.data.movieclips.remove(clip)

    def test_a_camera_filmed_at_another_frame_rate_stays_in_step_with_camera_1(self) -> None:
        scene = bpy.context.scene
        frame_before = scene.frame_current
        with tempfile.TemporaryDirectory() as folder:
            video_path = write_moving_shapes_video(Path(folder) / "webcam.mp4", frame_count=120, frame_rate=29.0)
            clip = load_video_clip(video_path, started_later_seconds=-0.51, first_camera_frame_rate=30.0)
            try:
                shown_frames = []
                for scene_frame in (1, 31, 91):
                    scene.frame_set(scene_frame)
                    shown_frames.append(scene_frame - clip.frame_start + 1)
                expected_frames = [round(((scene_frame - 1) / 30.0 + 0.51) * clip.fps) + 1 for scene_frame in (1, 31, 91)]
                self.assertEqual(shown_frames, expected_frames)
                static_clip = load_video_clip(video_path)
                self.assertIsNone(static_clip.animation_data.drivers.find("frame_start"))
                self.assertEqual(static_clip.frame_start, 1)
            finally:
                scene.frame_set(frame_before)
                bpy.data.movieclips.remove(clip)

    def test_set_buttons_take_the_current_frame_and_keep_start_before_end(self) -> None:
        scene = bpy.context.scene
        properties = scene.freeclipmocap
        properties.video_path = "//picked.mp4"
        frame_before = scene.frame_current
        try:
            scene.frame_current = 40
            self.assertEqual(bpy.ops.freeclipmocap.set_video_frame(edge="START"), {"FINISHED"})
            self.assertEqual((properties.video_start_frame, properties.video_end_frame), (40, 0))
            scene.frame_current = 25
            bpy.ops.freeclipmocap.set_video_frame(edge="END")
            self.assertEqual((properties.video_start_frame, properties.video_end_frame), (25, 25))
            properties.video_path = "//other.mp4"
            self.assertEqual((properties.video_start_frame, properties.video_end_frame), (1, 0))
        finally:
            properties.video_path = ""
            scene.frame_current = frame_before


if __name__ == "__main__":
    unittest.main()
