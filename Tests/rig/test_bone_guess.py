import unittest

from bl_ext.user_default.freeclipmocap.constants.bone_roles import BODY_ROLES, LEFT_FINGER_ROLES, REQUIRED_ROLES, Role
from bl_ext.user_default.freeclipmocap.rig.armature_bone_map import BUILT_IN_PRESETS_FOLDER
from bl_ext.user_default.freeclipmocap.rig.bone_guess import guess_bone_map, guess_from_names
from bl_ext.user_default.freeclipmocap.rig.bone_map_preset import load_presets

from .rig_fixtures import metarig_style_parents, mixamo_parents, unreal_style_parents


class GuessFromNamesTests(unittest.TestCase):
    def test_mixamo_rig_maps_every_role(self) -> None:
        bone_map = guess_from_names(mixamo_parents())
        self.assertEqual(len(bone_map.bones_by_role), 52)
        for role in BODY_ROLES + LEFT_FINGER_ROLES:
            self.assertEqual(bone_map.bone_for(role), f"mixamorig:{role.value}")

    def test_mixamo_rig_without_prefix(self) -> None:
        bone_map = guess_from_names(mixamo_parents(prefix=""))
        self.assertEqual(bone_map.bone_for(Role.LEFT_FORE_ARM), "LeftForeArm")
        self.assertEqual(bone_map.bone_for(Role.RIGHT_HAND_PINKY_3), "RightHandPinky3")
        self.assertEqual(bone_map.missing_required_roles(), [])

    def test_tips_extras_and_tails_stay_unmapped(self) -> None:
        mapped_bones = set(guess_from_names(mixamo_parents()).bones_by_role.values())
        for bone_name in ("mixamorig:HeadTop_End", "mixamorig:LeftToe_End", "mixamorig:LeftHandIndex4", "headfront"):
            self.assertNotIn(bone_name, mapped_bones)
        self.assertNotIn("Bone_007", mapped_bones)

    def test_unreal_style_rig(self) -> None:
        bone_map = guess_from_names(unreal_style_parents())
        self.assertEqual(bone_map.missing_required_roles(), [])
        self.assertEqual(bone_map.bone_for(Role.HIPS), "pelvis")
        self.assertEqual(
            [bone_map.bone_for(role) for role in (Role.SPINE, Role.SPINE_1, Role.SPINE_2)],
            ["spine_01", "spine_03", "spine_05"],
        )
        self.assertEqual(bone_map.bone_for(Role.NECK), "neck_01")
        self.assertEqual(bone_map.bone_for(Role.LEFT_SHOULDER), "clavicle_l")
        self.assertEqual(bone_map.bone_for(Role.LEFT_ARM), "upperarm_l")
        self.assertEqual(bone_map.bone_for(Role.RIGHT_LEG), "calf_r")
        self.assertEqual(bone_map.bone_for(Role.RIGHT_FOOT), "foot_r")
        self.assertEqual(bone_map.bone_for(Role.LEFT_TOE_BASE), "ball_l")
        self.assertEqual(bone_map.bone_for(Role.LEFT_HAND_INDEX_1), "index_01_l")
        self.assertEqual(bone_map.bone_for(Role.LEFT_HAND_THUMB_3), "thumb_03_l")

    def test_metarig_style_rig(self) -> None:
        bone_map = guess_from_names(metarig_style_parents())
        self.assertEqual(bone_map.missing_required_roles(), [Role.HIPS])
        self.assertEqual(bone_map.bone_for(Role.LEFT_ARM), "upper_arm.L")
        self.assertEqual(bone_map.bone_for(Role.RIGHT_LEG), "shin.R")
        self.assertEqual(bone_map.bone_for(Role.HEAD), "head")


class GuessBoneMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.presets, problems = load_presets(BUILT_IN_PRESETS_FOLDER, None)
        self.assertEqual(problems, [])

    def test_mixamo_preset_wins_on_a_mixamo_rig(self) -> None:
        guess = guess_bone_map(mixamo_parents(), self.presets)
        self.assertEqual(guess.preset_name, "Mixamo")
        self.assertEqual(len(guess.bone_map.bones_by_role), 52)

    def test_mixamo_preset_ignores_another_namespace(self) -> None:
        guess = guess_bone_map(mixamo_parents(prefix="mixamorig1:"), self.presets)
        self.assertEqual(guess.preset_name, "Mixamo")
        self.assertEqual(guess.bone_map.bone_for(Role.LEFT_UP_LEG), "mixamorig1:LeftUpLeg")

    def test_names_fill_what_no_preset_matches(self) -> None:
        guess = guess_bone_map(unreal_style_parents(), self.presets)
        self.assertIsNone(guess.preset_name)
        self.assertTrue(all(guess.bone_map.bone_for(role) for role in REQUIRED_ROLES))


if __name__ == "__main__":
    unittest.main()
