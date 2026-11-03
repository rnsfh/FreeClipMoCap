import unittest

import bpy

from bl_ext.user_default.freeclipmocap.constants.bone_roles import Role
from bl_ext.user_default.freeclipmocap.rig.armature_bone_map import has_bone_map, read_bone_map, user_presets_folder
from bl_ext.user_default.freeclipmocap.rig.bone_map_preset import preset_path

from .rig_fixtures import mixamo_parents


def create_armature(name: str, parents_by_bone: dict[str, str | None]) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new(name)
    armature_object = bpy.data.objects.new(name, armature_data)
    bpy.context.scene.collection.objects.link(armature_object)
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature_data.edit_bones
    for index, bone_name in enumerate(parents_by_bone):
        edit_bone = edit_bones.new(bone_name)
        edit_bone.head = (0.0, 0.0, index * 0.01)
        edit_bone.tail = (0.0, 0.0, index * 0.01 + 0.05)
    for bone_name, parent_name in parents_by_bone.items():
        if parent_name is not None:
            edit_bones[bone_name].parent = edit_bones[parent_name]
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature_object


class ArmatureBoneMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.armature_object = create_armature("SmartRigArmature", mixamo_parents())

    def tearDown(self) -> None:
        bpy.context.scene.freeclipmocap.target_armature = None
        armature_data = self.armature_object.data
        bpy.data.objects.remove(self.armature_object)
        bpy.data.armatures.remove(armature_data)

    def test_picking_an_armature_guesses_its_bone_map(self) -> None:
        self.assertFalse(has_bone_map(self.armature_object))
        bpy.context.scene.freeclipmocap.target_armature = self.armature_object
        bone_map = read_bone_map(self.armature_object)
        self.assertEqual(len(bone_map.bones_by_role), 52)
        self.assertEqual(bone_map.bone_for(Role.LEFT_FORE_ARM), "mixamorig:LeftForeArm")

    def test_picking_again_keeps_hand_edits(self) -> None:
        scene_properties = bpy.context.scene.freeclipmocap
        scene_properties.target_armature = self.armature_object
        entry = next(entry for entry in self.armature_object.freeclipmocap_rig.bone_map if entry.role == "Neck")
        entry.bone_name = ""
        scene_properties.target_armature = None
        scene_properties.target_armature = self.armature_object
        self.assertIsNone(read_bone_map(self.armature_object).bone_for(Role.NECK))

    def test_guess_operator_restores_the_map(self) -> None:
        bpy.context.scene.freeclipmocap.target_armature = self.armature_object
        for entry in self.armature_object.freeclipmocap_rig.bone_map:
            entry.bone_name = ""
        self.assertEqual(bpy.ops.freeclipmocap.guess_bone_map("EXEC_DEFAULT"), {"FINISHED"})
        self.assertEqual(len(read_bone_map(self.armature_object).bones_by_role), 52)

    def test_apply_preset_operator(self) -> None:
        scene_properties = bpy.context.scene.freeclipmocap
        scene_properties.target_armature = self.armature_object
        scene_properties.bone_map_preset = "Mixamo"
        self.assertEqual(bpy.ops.freeclipmocap.apply_bone_map_preset("EXEC_DEFAULT"), {"FINISHED"})
        self.assertEqual(read_bone_map(self.armature_object).missing_required_roles(), [])

    def test_save_and_delete_preset_operators(self) -> None:
        test_preset_name = "FreeClipMocap Test Preset"
        path = preset_path(user_presets_folder(), test_preset_name)
        if path.exists():
            self.skipTest(f"{path} exists already; left untouched.")
        scene_properties = bpy.context.scene.freeclipmocap
        scene_properties.target_armature = self.armature_object
        result = bpy.ops.freeclipmocap.save_bone_map_preset("EXEC_DEFAULT", preset_name=test_preset_name)
        self.assertEqual(result, {"FINISHED"})
        self.assertTrue(path.exists())
        self.assertEqual(scene_properties.bone_map_preset, test_preset_name)
        self.assertEqual(bpy.ops.freeclipmocap.delete_bone_map_preset("EXEC_DEFAULT"), {"FINISHED"})
        self.assertFalse(path.exists())
        scene_properties.bone_map_preset = "Mixamo"
        self.assertFalse(bpy.ops.freeclipmocap.delete_bone_map_preset.poll())

    def test_map_entries_naming_missing_bones_are_ignored(self) -> None:
        bpy.context.scene.freeclipmocap.target_armature = self.armature_object
        entry = next(entry for entry in self.armature_object.freeclipmocap_rig.bone_map if entry.role == "Hips")
        entry.bone_name = "renamed_away"
        self.assertEqual(read_bone_map(self.armature_object).missing_required_roles(), [Role.HIPS])


if __name__ == "__main__":
    unittest.main()
