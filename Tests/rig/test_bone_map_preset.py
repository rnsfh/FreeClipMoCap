import json
import tempfile
import unittest
from pathlib import Path

from bl_ext.user_default.freeclipmocap.constants.bone_roles import Role
from bl_ext.user_default.freeclipmocap.rig.bone_map import BoneMap
from bl_ext.user_default.freeclipmocap.rig.bone_map_preset import (
    BoneMapPreset,
    PresetError,
    is_valid_preset_name,
    load_presets,
)


class BoneMapPresetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_folder = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary_folder.name)

    def tearDown(self) -> None:
        self.temporary_folder.cleanup()

    def test_save_and_load(self) -> None:
        preset = BoneMapPreset(
            name="My Rig",
            bones_by_role={Role.HIPS: "pelvis", Role.LEFT_ARM: "upperarm_l"},
            is_built_in=False,
        )
        path = preset.save(self.folder)
        self.assertEqual(path.name, "My Rig.json")
        loaded = BoneMapPreset.load(path, is_built_in=False)
        self.assertEqual(loaded, preset)

    def test_rig_settings_are_saved_and_checked(self) -> None:
        preset = BoneMapPreset(
            name="Control Rig",
            bones_by_role={Role.HIPS: "torso"},
            is_built_in=False,
            rig_settings={"upper_arm_parent.L": {"IK_FK": 1.0}},
        )
        self.assertEqual(BoneMapPreset.load(preset.save(self.folder), is_built_in=False), preset)
        (self.folder / "Bad.json").write_text(
            json.dumps({"format_version": 1, "name": "Bad", "bones_by_role": {}, "rig_settings": {"torso": {"neck_follow": "off"}}}),
            encoding="utf-8",
        )
        with self.assertRaises(PresetError):
            BoneMapPreset.load(self.folder / "Bad.json", is_built_in=False)

    def test_delete_removes_only_the_presets_file(self) -> None:
        BoneMapPreset(name="Keep", bones_by_role={Role.HEAD: "head"}, is_built_in=False).save(self.folder)
        BoneMapPreset(name="Remove", bones_by_role={Role.HEAD: "head"}, is_built_in=False).save(self.folder)
        presets, _problems = load_presets(self.folder / "no built-in presets", self.folder)
        next(preset for preset in presets if preset.name == "Remove").delete()
        self.assertEqual(sorted(path.name for path in self.folder.iterdir()), ["Keep.json"])

    def test_built_in_preset_cannot_be_deleted(self) -> None:
        path = BoneMapPreset(name="Mixamo", bones_by_role={Role.HEAD: "head"}, is_built_in=True).save(self.folder)
        with self.assertRaises(PresetError):
            BoneMapPreset.load(path, is_built_in=True).delete()
        self.assertTrue(path.exists())

    def test_unknown_role_is_reported_and_other_presets_still_load(self) -> None:
        (self.folder / "Broken.json").write_text(
            json.dumps({"format_version": 1, "name": "Broken", "bones_by_role": {"Tail": "Bone_007"}}), encoding="utf-8"
        )
        BoneMapPreset(name="Good", bones_by_role={Role.HEAD: "head"}, is_built_in=False).save(self.folder)
        presets, problems = load_presets(self.folder, None)
        self.assertEqual([preset.name for preset in presets], ["Good"])
        self.assertEqual(len(problems), 1)
        self.assertIn("Tail", problems[0])

    def test_invalid_names_are_refused(self) -> None:
        self.assertTrue(is_valid_preset_name("Gnoll Rig_2"))
        for name in ("", "../escape", "a/b", "C:evil", " leading space"):
            self.assertFalse(is_valid_preset_name(name), name)
        with self.assertRaises(PresetError):
            BoneMapPreset(name="../escape", bones_by_role={}, is_built_in=False).save(self.folder)

    def test_match_needs_a_unique_short_name(self) -> None:
        preset = BoneMapPreset(name="Mixamo", bones_by_role={Role.HIPS: "mixamorig:Hips"}, is_built_in=True)
        self.assertEqual(preset.matched_to(["rigA:Hips"]).bone_for(Role.HIPS), "rigA:Hips")
        self.assertIsNone(preset.matched_to(["rigA:Hips", "rigB:Hips"]).bone_for(Role.HIPS))
        self.assertEqual(preset.matched_to(["rigA:Hips", "mixamorig:Hips"]).bone_for(Role.HIPS), "mixamorig:Hips")


class BoneMapTests(unittest.TestCase):
    def test_merge_keeps_own_roles_and_never_reuses_a_bone(self) -> None:
        own = BoneMap(bones_by_role={Role.HIPS: "pelvis"})
        fallback = BoneMap(bones_by_role={Role.HIPS: "root", Role.SPINE: "pelvis", Role.HEAD: "head"})
        merged = own.merged_with(fallback)
        self.assertEqual(dict(merged.bones_by_role), {Role.HIPS: "pelvis", Role.HEAD: "head"})


if __name__ == "__main__":
    unittest.main()
