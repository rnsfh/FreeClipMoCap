import unittest

import bpy
import numpy as np

from bl_ext.user_default.freeclipmocap.animation.action_writer import write_take
from bl_ext.user_default.freeclipmocap.animation.export import take_named_and_starting_at_zero
from bl_ext.user_default.freeclipmocap.animation.ik_controls import has_ik_controls, muted_ik_controls
from bl_ext.user_default.freeclipmocap.constants import pose_landmarks
from bl_ext.user_default.freeclipmocap.rig.armature_bone_map import guess_and_write_bone_map
from bl_ext.user_default.freeclipmocap.solver.mediapipe_table import MEDIAPIPE_TABLE
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolveSettings, solve

from ..animation.test_action_writer import create_armature, remove_armature
from ..solver.synthetic_actor import make_cache, make_rig, rest_landmarks, rotation_about

_FRAME_COUNT = 24
_LIMB_BONES = (
    "mixamorig:LeftUpLeg",
    "mixamorig:LeftLeg",
    "mixamorig:LeftFoot",
    "mixamorig:RightUpLeg",
    "mixamorig:RightLeg",
    "mixamorig:RightFoot",
    "mixamorig:LeftArm",
    "mixamorig:LeftForeArm",
    "mixamorig:LeftHand",
    "mixamorig:RightArm",
    "mixamorig:RightForeArm",
    "mixamorig:RightHand",
)


def bent_limbs(rest: dict[str, np.ndarray], fraction: float) -> dict[str, np.ndarray]:
    frame = {name: points.copy() for name, points in rest.items()}
    pose = frame["pose"]
    for side, elbow_sign in (("LEFT", -1.0), ("RIGHT", 1.0)):
        elbow = getattr(pose_landmarks, f"{side}_ELBOW")
        hand_points = [getattr(pose_landmarks, f"{side}_{name}") for name in ("WRIST", "PINKY", "INDEX", "THUMB")]
        bend = rotation_about((0.0, 0.0, 1.0), elbow_sign * 80.0 * fraction) @ rotation_about((0.0, 1.0, 0.0), 30.0 * fraction)
        pose[hand_points] = (pose[hand_points] - pose[elbow]) @ bend.T + pose[elbow]
        hand_name = side.title()
        frame[hand_name] = (frame[hand_name] - rest["pose"][elbow]) @ bend.T + pose[elbow]
        knee = getattr(pose_landmarks, f"{side}_KNEE")
        foot_points = [getattr(pose_landmarks, f"{side}_{name}") for name in ("ANKLE", "HEEL", "FOOT_INDEX")]
        kick = rotation_about((1.0, 0.0, 0.0), (50.0 if side == "LEFT" else 25.0) * fraction)
        pose[foot_points] = (pose[foot_points] - pose[knee]) @ kick.T + pose[knee]
    return frame


class IkControlsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = bpy.context.scene
        rig, bone_map = make_rig()
        bones = [
            (name, None if parent < 0 else rig.bone_names[parent], tuple(head), tuple(tail))
            for name, parent, head, tail in zip(rig.bone_names, rig.parent_indices, rig.heads, rig.tails)
        ]
        self.objects_before = set(bpy.data.objects)
        self.actions_before = set(bpy.data.actions)
        self.collections_before = set(bpy.data.collections)
        self.armature_object = create_armature("IkRig", bones)
        guess_and_write_bone_map(self.armature_object)
        rest = rest_landmarks(rig)
        frames = [bent_limbs(rest, fraction) for fraction in np.linspace(0.0, 1.0, _FRAME_COUNT)]
        take = solve(make_cache(frames), rig, bone_map, MEDIAPIPE_TABLE, SolveSettings(smoothing=0.0))
        write_take(self.armature_object, take, "IkTake", 1, None)
        self.scene.frame_start = 1
        self.scene.frame_end = _FRAME_COUNT
        self.scene.render.fps = 30
        self.scene.render.fps_base = 1.0
        self.scene.freeclipmocap.target_armature = self.armature_object
        self.reduce_keys_before = self.scene.freeclipmocap.reduce_keys

    def tearDown(self) -> None:
        self.scene.freeclipmocap.target_armature = None
        remove_armature(self.armature_object)
        for obj in set(bpy.data.objects) - self.objects_before:
            bpy.data.objects.remove(obj)
        for action in set(bpy.data.actions) - self.actions_before:
            bpy.data.actions.remove(action)
        for collection in set(bpy.data.collections) - self.collections_before:
            bpy.data.collections.remove(collection)
        for armature in [armature for armature in bpy.data.armatures if armature.users == 0]:
            bpy.data.armatures.remove(armature)
        self.scene.freeclipmocap.reduce_keys = self.reduce_keys_before

    def pose(self, frame: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.scene.frame_set(frame)
        pose_bones = self.armature_object.pose.bones
        matrices = {name: np.array(pose_bones[name].matrix.to_3x3().normalized()) for name in _LIMB_BONES}
        tails = {name: np.array(pose_bones[name].tail) for name in _LIMB_BONES}
        return matrices, tails

    def assert_same_pose(self, first: tuple[dict, dict], second: tuple[dict, dict], degrees: float, metres: float) -> None:
        for name in _LIMB_BONES:
            cosine = (np.trace(first[0][name].T @ second[0][name]) - 1.0) / 2.0
            self.assertLess(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), degrees, name)
            self.assertLess(float(np.abs(first[1][name] - second[1][name]).max()), metres, name)

    def rotation_keys(self, bone_name: str) -> np.ndarray:
        action = self.armature_object.animation_data.action
        keys = []
        for channel in range(4):
            fcurve = action.fcurve_ensure_for_datablock(
                self.armature_object, f'pose.bones["{bone_name}"].rotation_quaternion', index=channel
            )
            keys.append([tuple(point.co) for point in fcurve.keyframe_points])
        return np.array(keys)

    def control(self, name: str) -> bpy.types.Object:
        return bpy.data.objects[f"IkRig {name}"]

    def raise_target(self, name: str, metres: float) -> None:
        action = self.control(name).animation_data.action
        fcurve = action.fcurve_ensure_for_datablock(self.control(name), "location", index=2)
        for point in fcurve.keyframe_points:
            for attribute in ("co", "handle_left", "handle_right"):
                getattr(point, attribute)[1] += metres
        fcurve.update()

    def test_ik_plays_the_fk_take_until_a_target_is_moved(self) -> None:
        self.assertEqual(bpy.ops.freeclipmocap.add_ik_controls(), {"FINISHED"})
        self.assertTrue(has_ik_controls(self.armature_object))
        for limb in ("Left Leg", "Right Leg", "Left Arm", "Right Arm"):
            self.assertEqual(self.armature_object[f"IK {limb}"], 1.0)
            self.assertEqual(self.control(f"Pole {limb}").parent, self.armature_object)
        for frame in (1, 8, 16, _FRAME_COUNT):
            ik_pose = self.pose(frame)
            with muted_ik_controls(self.armature_object):
                fk_pose = self.pose(frame)
            self.assert_same_pose(ik_pose, fk_pose, degrees=0.5, metres=0.002)

        self.raise_target("IK Left Leg", 0.05)
        frame = 16
        self.scene.frame_set(frame)
        target = np.array(self.armature_object.matrix_world.inverted() @ self.control("IK Left Leg").matrix_world.translation)
        shin_tail = np.array(self.armature_object.pose.bones["mixamorig:LeftLeg"].tail)
        self.assertLess(float(np.abs(shin_tail - target).max()), 0.002)
        with muted_ik_controls(self.armature_object):
            fk_pose = self.pose(frame)
        self.assertGreater(float(np.abs(shin_tail - fk_pose[1]["mixamorig:LeftLeg"]).max()), 0.03)

        self.armature_object["IK Left Leg"] = 0.0
        self.assert_same_pose(self.pose(frame), fk_pose, degrees=0.01, metres=1e-5)

    def test_sliders_are_keyable(self) -> None:
        bpy.ops.freeclipmocap.add_ik_controls()
        self.raise_target("IK Left Arm", 0.05)
        self.armature_object["IK Left Arm"] = 0.0
        self.armature_object.keyframe_insert('["IK Left Arm"]', frame=1)
        self.armature_object["IK Left Arm"] = 1.0
        self.armature_object.keyframe_insert('["IK Left Arm"]', frame=_FRAME_COUNT)
        self.scene.frame_set(1)
        forearm_constraint = self.armature_object.pose.bones["mixamorig:LeftForeArm"].constraints["FreeClipMocap IK"]
        self.assertAlmostEqual(forearm_constraint.influence, 0.0, places=5)
        self.scene.frame_set(_FRAME_COUNT)
        self.assertAlmostEqual(forearm_constraint.influence, 1.0, places=5)

    def test_removing_the_controls_can_keep_the_ik_pose_in_the_fk_keys(self) -> None:
        self.scene.freeclipmocap.reduce_keys = False
        left_arm_keys_before = self.rotation_keys("mixamorig:LeftForeArm")
        bpy.ops.freeclipmocap.add_ik_controls()
        self.raise_target("IK Right Arm", 0.04)
        ik_pose = self.pose(12)

        self.assertEqual(bpy.ops.freeclipmocap.remove_ik_controls(keep_pose=True), {"FINISHED"})

        self.assertFalse(has_ik_controls(self.armature_object))
        self.assertFalse(any(name.startswith("IkRig IK") or name.startswith("IkRig Pole") for name in bpy.data.objects.keys()))
        self.assertNotIn("IK Right Arm", self.armature_object.keys())
        self.assertNotIn("IkRig IK Controls", bpy.data.collections)
        self.assertFalse(any(constraint for bone in self.armature_object.pose.bones for constraint in bone.constraints))
        self.assert_same_pose(self.pose(12), ik_pose, degrees=0.05, metres=0.0005)
        np.testing.assert_array_equal(self.rotation_keys("mixamorig:LeftForeArm"), left_arm_keys_before)

    def test_removing_without_keeping_returns_to_the_fk_take(self) -> None:
        with_fk_only = self.pose(12)
        bpy.ops.freeclipmocap.add_ik_controls()
        self.raise_target("IK Right Arm", 0.04)

        bpy.ops.freeclipmocap.remove_ik_controls(keep_pose=False)

        self.assert_same_pose(self.pose(12), with_fk_only, degrees=0.01, metres=1e-5)

    def test_export_shifts_the_ik_targets_with_the_take(self) -> None:
        action = self.armature_object.animation_data.action
        for fcurve in (fcurve for layer in action.layers for strip in layer.strips for bag in strip.channelbags for fcurve in bag.fcurves):
            for point in fcurve.keyframe_points:
                for attribute in ("co", "handle_left", "handle_right"):
                    getattr(point, attribute)[0] += 4
            fcurve.update()
        self.scene.frame_start = 5
        self.scene.frame_end = 5 + _FRAME_COUNT - 1
        bpy.ops.freeclipmocap.add_ik_controls()
        self.raise_target("IK Left Leg", 0.05)
        played = self.pose(5 + 10)

        with take_named_and_starting_at_zero(self.scene, self.armature_object):
            self.assertEqual(self.scene.frame_start, 0)
            exported = self.pose(10)

        # Blender's IK starts from the pose it solved last, so the same frame can differ by a few hundredths of a degree.
        self.assert_same_pose(exported, played, degrees=0.1, metres=1e-4)
        self.assert_same_pose(self.pose(5 + 10), played, degrees=0.1, metres=1e-4)


if __name__ == "__main__":
    unittest.main()
