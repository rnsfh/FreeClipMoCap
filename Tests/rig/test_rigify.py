import unittest

import addon_utils
import bpy
import numpy as np

from bl_ext.user_default.freeclipmocap.animation.action_writer import write_take
from bl_ext.user_default.freeclipmocap.rig.armature_bone_map import (
    apply_rig_settings,
    guess_and_write_bone_map,
    read_bone_map,
)
from bl_ext.user_default.freeclipmocap.rig.armature_rest import rig_rest_from_armature
from bl_ext.user_default.freeclipmocap.solver.mediapipe_table import MEDIAPIPE_TABLE
from bl_ext.user_default.freeclipmocap.solver.pose_solver import SolveSettings, solve

from ..solver.synthetic_actor import make_cache, make_rig, rest_landmarks, rotation_about

_RIGIFY = "rigify"


def bend_degrees(rig_object: bpy.types.Object, upper: str, lower: str) -> float:
    upper_axis = rig_object.pose.bones[upper].matrix.to_3x3().col[1]
    lower_axis = rig_object.pose.bones[lower].matrix.to_3x3().col[1]
    return float(np.degrees(upper_axis.angle(lower_axis)))


class RigifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rigify_was_enabled = addon_utils.check(_RIGIFY)[1]
        if not cls.rigify_was_enabled:
            bpy.ops.preferences.addon_enable(module=_RIGIFY)
        cls.objects_before = set(bpy.data.objects)
        cls.armatures_before = set(bpy.data.armatures)
        cls.actions_before = set(bpy.data.actions)
        bpy.ops.object.armature_human_metarig_add()
        metarig = bpy.context.active_object
        bpy.ops.pose.rigify_generate()
        cls.rig_object = next(obj for obj in set(bpy.data.objects) - cls.objects_before if obj.type == "ARMATURE" and obj is not metarig)

    @classmethod
    def tearDownClass(cls) -> None:
        for obj in set(bpy.data.objects) - cls.objects_before:
            bpy.data.objects.remove(obj)
        for armature in set(bpy.data.armatures) - cls.armatures_before:
            bpy.data.armatures.remove(armature)
        for action in set(bpy.data.actions) - cls.actions_before:
            bpy.data.actions.remove(action)
        if not cls.rigify_was_enabled:
            bpy.ops.preferences.addon_disable(module=_RIGIFY)

    def test_rigify_rig_is_mapped_switched_to_fk_and_follows_the_take_exactly(self) -> None:
        guess = guess_and_write_bone_map(self.rig_object)
        self.assertEqual(guess.preset_name, "Rigify")
        bone_map = read_bone_map(self.rig_object)
        self.assertEqual(len(bone_map.bones_by_role), 52)
        self.assertFalse(bone_map.missing_required_roles())

        changes = apply_rig_settings(self.rig_object)
        self.assertIn("upper_arm_parent.L IK_FK 0 -> 1", changes)
        self.assertEqual(self.rig_object.pose.bones["torso"]["neck_follow"], 0.0)
        self.assertEqual(apply_rig_settings(self.rig_object), [])

        synthetic_rig, _synthetic_map = make_rig()
        rest = rest_landmarks(synthetic_rig)
        elbow = rest["pose"][13]
        bend = rotation_about((0.0, 0.0, 1.0), -90.0)
        bent = {name: points.copy() for name, points in rest.items()}
        bent["pose"][[15, 17, 19, 21]] = (rest["pose"][[15, 17, 19, 21]] - elbow) @ bend.T + elbow
        bent["Left"] = (rest["Left"] - elbow) @ bend.T + elbow
        take = solve(make_cache([rest] * 3 + [bent] * 3), rig_rest_from_armature(self.rig_object), bone_map, MEDIAPIPE_TABLE, SolveSettings(0.0))
        written = write_take(self.rig_object, take, "RigifyTake", 1, None)
        self.assertEqual(written.unmatched_bones, ())

        bpy.context.scene.frame_set(1)
        self.assertLess(bend_degrees(self.rig_object, "ORG-upper_arm.L", "ORG-forearm.L"), 1.0)
        bpy.context.scene.frame_set(6)
        self.assertAlmostEqual(bend_degrees(self.rig_object, "ORG-upper_arm.L", "ORG-forearm.L"), 90.0, delta=1.0)
        self.assertLess(bend_degrees(self.rig_object, "ORG-upper_arm.R", "ORG-forearm.R"), 1.0)


if __name__ == "__main__":
    unittest.main()
