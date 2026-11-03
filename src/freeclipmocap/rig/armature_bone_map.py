import json
from pathlib import Path

import bpy

from ..constants.bone_roles import BODY_ROLES, LEFT_FINGER_ROLES, RIGHT_FINGER_ROLES, Role
from .bone_guess import BoneGuess, guess_bone_map
from .bone_map import BoneMap
from .bone_map_preset import BoneMapPreset, load_presets, read_rig_settings

ALL_ROLES = BODY_ROLES + LEFT_FINGER_ROLES + RIGHT_FINGER_ROLES
BUILT_IN_PRESETS_FOLDER = Path(__file__).resolve().parent.parent / "presets" / "bone_maps"
_EXTENSION_PACKAGE = __name__.rsplit(".", 2)[0]
_USER_PRESETS_FOLDER_NAME = "bone_map_presets"


def user_presets_folder() -> Path:
    return Path(bpy.utils.extension_path_user(_EXTENSION_PACKAGE, path=_USER_PRESETS_FOLDER_NAME, create=True))


def available_presets() -> tuple[list[BoneMapPreset], list[str]]:
    return load_presets(BUILT_IN_PRESETS_FOLDER, user_presets_folder())


def bone_parents(armature_object: bpy.types.Object) -> dict[str, str | None]:
    return {
        bone.name: bone.parent.name if bone.parent is not None else None
        for bone in armature_object.data.bones
    }


def read_bone_map(armature_object: bpy.types.Object) -> BoneMap:
    bones_by_role: dict[Role, str] = {}
    bone_names = set(armature_object.data.bones.keys())
    for entry in armature_object.freeclipmocap_rig.bone_map:
        if entry.bone_name and entry.bone_name in bone_names:
            bones_by_role[Role(entry.role)] = entry.bone_name
    return BoneMap(bones_by_role=bones_by_role)


def write_bone_map(armature_object: bpy.types.Object, bone_map: BoneMap) -> None:
    entries = armature_object.freeclipmocap_rig.bone_map
    entries.clear()
    for role in ALL_ROLES:
        entry = entries.add()
        entry.role = role.value
        entry.bone_name = bone_map.bone_for(role) or ""


def has_bone_map(armature_object: bpy.types.Object) -> bool:
    return any(entry.bone_name for entry in armature_object.freeclipmocap_rig.bone_map)


def guess_and_write_bone_map(armature_object: bpy.types.Object) -> BoneGuess:
    presets, _problems = available_presets()
    guess = guess_bone_map(bone_parents(armature_object), presets)
    write_bone_map(armature_object, guess.bone_map)
    write_rig_settings(armature_object, next((preset for preset in presets if preset.name == guess.preset_name), None))
    return guess


def write_rig_settings(armature_object: bpy.types.Object, preset: BoneMapPreset | None) -> None:
    settings = preset.rig_settings if preset is not None else {}
    armature_object.freeclipmocap_rig.rig_settings_json = json.dumps(settings, sort_keys=True) if settings else ""


def read_armature_rig_settings(armature_object: bpy.types.Object) -> dict[str, dict[str, float]]:
    text = armature_object.freeclipmocap_rig.rig_settings_json
    return read_rig_settings(json.loads(text), armature_object.name) if text else {}


def apply_rig_settings(armature_object: bpy.types.Object) -> list[str]:
    changes: list[str] = []
    for bone_name, values in read_armature_rig_settings(armature_object).items():
        pose_bone = armature_object.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        for setting, value in values.items():
            if setting in pose_bone and float(pose_bone[setting]) != value:
                changes.append(f"{bone_name} {setting} {float(pose_bone[setting]):g} -> {value:g}")
                pose_bone[setting] = value
    return changes
