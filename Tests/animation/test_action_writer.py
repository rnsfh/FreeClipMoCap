import tempfile
import unittest
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion

from bl_ext.user_default.freeclipmocap.animation.action_writer import key_tolerances, write_take
from bl_ext.user_default.freeclipmocap.capture.landmark_cache import LandmarkCache, VideoSource, cache_path_for
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolvedTake

from ..solver.synthetic_actor import make_cache, make_rig, rest_landmarks, rotation_about


def create_armature(name: str, bones: list[tuple[str, str | None, tuple[float, float, float], tuple[float, float, float]]]) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new(name)
    armature_object = bpy.data.objects.new(name, armature_data)
    bpy.context.scene.collection.objects.link(armature_object)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")
    for bone_name, parent_name, head, tail in bones:
        edit_bone = armature_data.edit_bones.new(bone_name)
        edit_bone.head = head
        edit_bone.tail = tail
        if parent_name is not None:
            edit_bone.parent = armature_data.edit_bones[parent_name]
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature_object


def remove_armature(armature_object: bpy.types.Object) -> None:
    armature_data = armature_object.data
    bpy.data.objects.remove(armature_object)
    bpy.data.armatures.remove(armature_data)


class ActionWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.armature_object = create_armature(
            "WriterRig",
            [("Root", None, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), ("Child", "Root", (0.0, 0.0, 1.0), (0.0, 0.0, 2.0))],
        )
        self.armature_object.pose.bones["Child"].rotation_mode = "XYZ"
        angles = np.radians([0.0, 10.0, 20.0, 30.0])
        quaternions = np.stack((np.cos(angles / 2), np.sin(angles / 2), np.zeros(4), np.zeros(4)), axis=-1)
        self.take = SolvedTake(
            times_seconds=np.arange(4) / 24.0,
            rotations_by_bone={"Root": quaternions, "Child": quaternions},
            warnings=(),
            camera_tilt_degrees=0.0,
        )
        self.actions_before = set(bpy.data.actions)

    def tearDown(self) -> None:
        remove_armature(self.armature_object)
        for action in set(bpy.data.actions) - self.actions_before:
            bpy.data.actions.remove(action)

    def test_quaternion_and_euler_bones_are_keyed_in_their_own_mode(self) -> None:
        action = write_take(self.armature_object, self.take, "Take", start_frame=5, key_tolerance_degrees=None).action
        self.assertTrue(action.use_fake_user)
        self.assertIs(self.armature_object.animation_data.action, action)
        bpy.context.scene.frame_set(8)
        root = self.armature_object.pose.bones["Root"]
        child = self.armature_object.pose.bones["Child"]
        self.assertAlmostEqual(root.rotation_quaternion.to_euler().x, np.radians(30.0), places=5)
        self.assertAlmostEqual(child.rotation_euler.x, np.radians(30.0), places=5)
        self.assertEqual(tuple(child.rotation_quaternion), (1.0, 0.0, 0.0, 0.0))

    def test_reduced_keys_stay_within_the_tolerance(self) -> None:
        frame_count = 120
        angles = np.radians(60.0) * np.sin(np.linspace(0.0, 2.0 * np.pi, frame_count)) + np.radians(
            3.0
        ) * np.random.default_rng(5).normal(size=frame_count)
        quaternions = np.stack((np.cos(angles / 2), np.sin(angles / 2), np.zeros(frame_count), np.zeros(frame_count)), axis=-1)
        take = SolvedTake(
            times_seconds=np.arange(frame_count) / 30.0,
            rotations_by_bone={"Root": quaternions, "Child": quaternions},
            warnings=(),
            camera_tilt_degrees=0.0,
        )
        tolerance_degrees = 1.0
        action = write_take(self.armature_object, take, "Reduced", start_frame=1, key_tolerance_degrees=tolerance_degrees).action
        for bone_name, data_path in (("Root", "rotation_quaternion"), ("Child", "rotation_euler")):
            curve = action.fcurve_ensure_for_datablock(self.armature_object, f'pose.bones["{bone_name}"].{data_path}', index=1 if data_path == "rotation_quaternion" else 0)
            self.assertLess(len(curve.keyframe_points), frame_count)
        for frame_index in range(frame_count):
            bpy.context.scene.frame_set(frame_index + 1)
            root_angle = 2.0 * np.arctan2(self.armature_object.pose.bones["Root"].rotation_quaternion.x, self.armature_object.pose.bones["Root"].rotation_quaternion.w)
            child_angle = self.armature_object.pose.bones["Child"].rotation_euler.x
            self.assertLess(abs(np.degrees(root_angle - angles[frame_index])), tolerance_degrees + 0.01)
            self.assertLess(abs(np.degrees(child_angle - angles[frame_index])), tolerance_degrees + 0.01)

    def test_deformed_meshes_are_left_as_they_were(self) -> None:
        mesh_data = bpy.data.meshes.new("Skin")
        mesh_data.from_pydata([(0.0, 0.0, 0.5), (0.0, 0.0, 1.5), (0.1, 0.0, 1.0)], [], [(0, 1, 2)])
        skin = bpy.data.objects.new("Skin", mesh_data)
        bpy.context.scene.collection.objects.link(skin)
        modifier = skin.modifiers.new("Armature", "ARMATURE")
        modifier.object = self.armature_object
        try:
            write_take(self.armature_object, self.take, "Take", start_frame=1, key_tolerance_degrees=None)
            self.assertFalse(skin.hide_viewport)
            self.assertTrue(modifier.show_viewport)
        finally:
            bpy.data.objects.remove(skin)
            bpy.data.meshes.remove(mesh_data)

    def test_every_solve_is_a_new_take(self) -> None:
        first = write_take(self.armature_object, self.take, "Take", start_frame=1, key_tolerance_degrees=None).action
        second = write_take(self.armature_object, self.take, "Take", start_frame=1, key_tolerance_degrees=None).action
        self.assertEqual((first.name, second.name), ("Take", "Take.001"))
        self.assertIs(self.armature_object.animation_data.action, second)


class KeyToleranceTests(unittest.TestCase):
    def test_bones_that_move_a_foot_get_a_tolerance_scaled_to_their_lever(self) -> None:
        rig, _bone_map = make_rig()
        bone_names = ["mixamorig:Hips", "mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot", "mixamorig:LeftArm"]
        tolerances = key_tolerances(rig, bone_names, ("mixamorig:LeftFoot", "mixamorig:RightFoot"), 2.0)
        self.assertEqual(tolerances["mixamorig:LeftArm"], 2.0)
        self.assertLess(tolerances["mixamorig:Hips"], tolerances["mixamorig:LeftUpLeg"])
        self.assertLess(tolerances["mixamorig:LeftUpLeg"], tolerances["mixamorig:LeftLeg"])
        self.assertLess(tolerances["mixamorig:LeftLeg"], tolerances["mixamorig:LeftFoot"])
        thigh_lever = float(np.linalg.norm(rig.tails[rig.index_of("mixamorig:LeftFoot")] - rig.heads[rig.index_of("mixamorig:LeftUpLeg")]))
        hips_lever = max(
            float(np.linalg.norm(rig.tails[rig.index_of(f"mixamorig:{side}Foot")] - rig.heads[rig.index_of("mixamorig:Hips")]))
            for side in ("Left", "Right")
        )
        self.assertAlmostEqual(
            np.radians(tolerances["mixamorig:LeftUpLeg"]) * thigh_lever, 0.002 * hips_lever, delta=0.0001
        )
        self.assertEqual(key_tolerances(rig, bone_names, (), 2.0)["mixamorig:Hips"], 2.0)


class SolveOperatorTests(unittest.TestCase):
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
        cache = LandmarkCache(
            tracking_model=synthetic.tracking_model,
            settings=synthetic.settings,
            video=VideoSource(status.st_size, status.st_mtime_ns, 30.0, 640, 480),
            frame_indices=synthetic.frame_indices,
            timestamps_milliseconds=synthetic.timestamps_milliseconds,
            pose=synthetic.pose,
            left_hand=synthetic.left_hand,
            right_hand=synthetic.right_hand,
        )
        cache.save(cache_path_for(self.video_path, cache.settings))
        properties = bpy.context.scene.freeclipmocap
        properties.video_path = str(self.video_path)
        properties.pose_model_variant = cache.settings.pose_model_variant.value
        properties.smoothing = 0.0
        properties.target_armature = self.armature_object
        bpy.context.scene.render.fps = 24
        bpy.context.scene.render.fps_base = 1.0
        bpy.context.scene.frame_start = 1
        self.actions_before = set(bpy.data.actions)

    def tearDown(self) -> None:
        bpy.context.scene.freeclipmocap.target_armature = None
        remove_armature(self.armature_object)
        for action in set(bpy.data.actions) - self.actions_before:
            bpy.data.actions.remove(action)
        self.temporary_folder.cleanup()

    def test_solve_writes_a_take_that_bends_the_elbow(self) -> None:
        self.assertEqual(bpy.ops.freeclipmocap.solve(), {"FINISHED"})
        action = self.armature_object.animation_data.action
        self.assertEqual(action.name, "clip")
        self.assertIs(bpy.context.view_layer.objects.active, self.armature_object)
        self.assertTrue(self.armature_object.select_get())
        scene = bpy.context.scene
        self.assertEqual((scene.render.fps, scene.render.fps_base), (30, 1.0))
        self.assertEqual(scene.frame_end, 31)
        hips_curve = action.fcurve_ensure_for_datablock(
            self.armature_object, 'pose.bones["mixamorig:Hips"].rotation_quaternion', index=0
        )
        self.assertEqual(len(hips_curve.keyframe_points), 2)
        hips_location_curve = action.fcurve_ensure_for_datablock(
            self.armature_object, 'pose.bones["mixamorig:Hips"].location', index=2
        )
        self.assertGreaterEqual(len(hips_location_curve.keyframe_points), 2)
        scene.frame_set(5)
        forearm = self.armature_object.pose.bones["mixamorig:LeftForeArm"]
        self.assertLess(Quaternion(forearm.rotation_quaternion).angle, np.radians(0.6))
        scene.frame_set(31)
        self.assertAlmostEqual(np.degrees(Quaternion(forearm.rotation_quaternion).angle), 90.0, delta=0.5)
        wrist = self.armature_object.matrix_world @ self.armature_object.pose.bones["mixamorig:LeftHand"].head
        elbow = self.armature_object.matrix_world @ forearm.head
        self.assertLess((wrist - elbow).normalized().y, -0.99)

    def test_solve_uses_only_the_chosen_video_frames_and_keeps_them_under_the_video(self) -> None:
        properties = bpy.context.scene.freeclipmocap
        properties.video_start_frame = 16
        properties.video_end_frame = 31
        self.assertEqual(bpy.ops.freeclipmocap.solve(), {"FINISHED"})
        scene = bpy.context.scene
        self.assertEqual((scene.frame_start, scene.frame_end), (16, 31))
        action = self.armature_object.animation_data.action
        hips_curve = action.fcurve_ensure_for_datablock(
            self.armature_object, 'pose.bones["mixamorig:Hips"].rotation_quaternion', index=0
        )
        self.assertEqual(hips_curve.keyframe_points[0].co.x, 16.0)
        forearm = self.armature_object.pose.bones["mixamorig:LeftForeArm"]
        for frame in (16, 31):
            scene.frame_set(frame)
            self.assertAlmostEqual(np.degrees(Quaternion(forearm.rotation_quaternion).angle), 90.0, delta=0.5)

    def test_start_after_end_is_refused(self) -> None:
        properties = bpy.context.scene.freeclipmocap
        properties.video_start_frame = 20
        properties.video_end_frame = 10
        with self.assertRaises(RuntimeError):
            bpy.ops.freeclipmocap.solve()


if __name__ == "__main__":
    unittest.main()
